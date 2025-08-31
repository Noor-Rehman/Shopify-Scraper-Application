"""
Optimized Flask Shopify Scraper with Real-time Logging
Features:
- Fast concurrent scraping with ThreadPoolExecutor
- Real-time log streaming to frontend
- Optimized download performance
- Simple, efficient approach
- Visual progress indicators with emoji logs
"""

import os
import time
import json
import uuid
import shutil
import threading
import requests
import urllib3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, render_template_string, jsonify, send_from_directory
from bs4 import BeautifulSoup
from PIL import Image, ImageFilter, ImageEnhance

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Optional dependencies
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("RemBG not available. Install with: pip install rembg")

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent.absolute()
DOWNLOADS_DIR = BASE_DIR / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

# Global job storage
JOBS = {}
STOP_FLAGS = {}
JOBS_LOCK = threading.Lock()

# ----------------------- Core Scraping Functions -----------------------

def log_message(job_id, message, log_type="info"):
    """Add real-time log message"""
    with JOBS_LOCK:
        if job_id in JOBS:
            timestamp = time.strftime('%H:%M:%S')
            emoji = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}.get(log_type, "")
            log_entry = f"[{timestamp}] {emoji} {message}"
            JOBS[job_id]['logs'].insert(0, log_entry)
            JOBS[job_id]['logs'] = JOBS[job_id]['logs'][:100]  # Keep last 100
            print(log_entry)  # Also print to console

def fetch_product_urls(store_url, max_products=None):
    """Fast product URL fetching using requests"""
    try:
        parsed_url = requests.utils.urlparse(store_url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        sitemap_url = f"{domain}/sitemap_products_1.xml?from=1878192586821&to=7199711461456"
        
        resp = requests.get(sitemap_url, verify=False, timeout=10)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.content, "xml")
        product_urls = [loc.text for loc in soup.find_all("loc") if "/products/" in loc.text]
        
        if max_products:
            product_urls = product_urls[:max_products]
        
        return product_urls, None
    except Exception as e:
        return [], str(e)

def scrape_product_images(url):
    """Fast product image scraping with structured data focus"""
    try:
        page = requests.get(url, verify=False, timeout=10)
        page.raise_for_status()
        
        soup = BeautifulSoup(page.content, "html.parser")
        img_urls = set()

        # Primary: JSON-LD structured data (fastest for Shopify)
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and "image" in data:
                    images = data["image"] if isinstance(data["image"], list) else [data["image"]]
                    img_urls.update(images)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "image" in item:
                            images = item["image"] if isinstance(item["image"], list) else [item["image"]]
                            img_urls.update(images)
            except:
                continue

        # Fallback: og:image and img tags
        if not img_urls:
            og_img = soup.find('meta', property='og:image')
            if og_img and og_img.get('content'):
                img_urls.add(og_img['content'])
            
            for img in soup.find_all('img'):
                src = img.get('src') or img.get('data-src')
                if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    if src.startswith('//'):
                        src = 'https:' + src
                    img_urls.add(src)

        return list(img_urls)
    except Exception as e:
        return []

def download_single_image(args):
    """Download single image with timing"""
    img_url, save_dir, job_id = args
    
    if STOP_FLAGS.get(job_id, False):
        return {"url": img_url, "status": "stopped", "filename": None, "elapsed": 0}
    
    try:
        filename = img_url.split("/")[-1].split("?")[0] or f"image_{int(time.time())}.jpg"
        filepath = save_dir / filename
        
        start = time.time()
        response = requests.get(img_url, verify=False, timeout=15, stream=True)
        response.raise_for_status()
        
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Create thumbnail for faster preview
        try:
            with Image.open(filepath) as img:
                img.thumbnail((300, 300), Image.LANCZOS)
                img.save(filepath, quality=85, optimize=True)
        except:
            pass
        
        elapsed = time.time() - start
        log_message(job_id, f"Downloaded {filename} in {elapsed:.2f}s" + (" (slow)" if elapsed > 2 else ""), 
                   "success" if elapsed <= 2 else "warning")
        
        return {"url": img_url, "status": "success", "filename": filename, "elapsed": elapsed}
        
    except Exception as e:
        log_message(job_id, f"Failed to download {img_url}: {str(e)}", "error")
        return {"url": img_url, "status": "error", "filename": None, "elapsed": 0, "error": str(e)}

def run_scraping_job(job_id):
    """Main scraping workflow with concurrent execution"""
    meta = JOBS[job_id]
    
    try:
        log_message(job_id, "Starting product discovery...", "info")
        meta['state'] = 'discovering'
        
        # Step 1: Get product URLs
        store_url = meta['store_url']
        max_products = meta['options'].get('max_products', 30)
        
        product_urls, error = fetch_product_urls(store_url, max_products)
        if error:
            raise Exception(f"Product discovery failed: {error}")
        
        if not product_urls:
            raise Exception("No products found in sitemap")
        
        log_message(job_id, f"Found {len(product_urls)} product pages", "success")
        meta['stats']['products'] = len(product_urls)
        
        if STOP_FLAGS.get(job_id, False):
            meta['state'] = 'stopped'
            return
        
        # Step 2: Scrape all image URLs
        log_message(job_id, "Discovering product images...", "info")
        meta['state'] = 'scraping_images'
        
        all_images = []
        max_workers = min(meta['options'].get('max_workers', 8), 12)
        
        def scrape_single_product(url):
            if STOP_FLAGS.get(job_id, False):
                return []
            images = scrape_product_images(url)
            if images:
                log_message(job_id, f"Found {len(images)} images in {url.split('/')[-1]}", "info")
            return images
        
        # Parallel product scraping
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(scrape_single_product, url): url for url in product_urls}
            
            for future in as_completed(future_to_url):
                if STOP_FLAGS.get(job_id, False):
                    break
                try:
                    images = future.result()
                    all_images.extend(images)
                except Exception as e:
                    url = future_to_url[future]
                    log_message(job_id, f"Error scraping {url}: {e}", "error")
                    meta['stats']['errors'] += 1
        
        # Deduplicate images
        unique_images = list(set(all_images))
        log_message(job_id, f"Found {len(unique_images)} unique images (removed {len(all_images) - len(unique_images)} duplicates)", "success")
        meta['stats']['images'] = len(unique_images)
        
        if not unique_images:
            raise Exception("No images found across all products")
        
        if STOP_FLAGS.get(job_id, False):
            meta['state'] = 'stopped'
            return
        
        # Step 3: Download images in parallel
        log_message(job_id, "Starting parallel downloads...", "info")
        meta['state'] = 'downloading'
        
        save_dir = Path(meta['folder'])
        download_args = [(img_url, save_dir, job_id) for img_url in unique_images]
        
        results = []
        download_workers = min(meta['options'].get('max_workers', 8), 15)
        
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            future_to_img = {executor.submit(download_single_image, args): args[0] for args in download_args}
            
            for future in as_completed(future_to_img):
                if STOP_FLAGS.get(job_id, False):
                    break
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['status'] == 'success':
                        meta['stats']['downloaded'] += 1
                    else:
                        meta['stats']['errors'] += 1
                        
                    # Update progress
                    progress = (len(results) / len(unique_images)) * 100
                    log_message(job_id, f"Progress: {len(results)}/{len(unique_images)} ({progress:.1f}%)", "info")
                    
                except Exception as e:
                    img_url = future_to_img[future]
                    log_message(job_id, f"Download error for {img_url}: {e}", "error")
                    meta['stats']['errors'] += 1
        
        # Calculate average download time
        successful_downloads = [r for r in results if r['status'] == 'success']
        if successful_downloads:
            avg_time = sum(r['elapsed'] for r in successful_downloads) / len(successful_downloads)
            meta['avg_time'] = avg_time
            log_message(job_id, f"Average download time: {avg_time:.2f}s", "success")
        
        # Store results
        meta['images'] = [
            {
                'url': r['url'],
                'filename': r.get('filename'),
                'elapsed': r.get('elapsed'),
                'error': r.get('error'),
                'processed_filename': None
            }
            for r in results
        ]
        
        if STOP_FLAGS.get(job_id, False):
            meta['state'] = 'stopped'
            return
        
        # Step 4: Create ZIP archive
        log_message(job_id, "Creating ZIP archive...", "info")
        zip_path = DOWNLOADS_DIR / f"{job_id}.zip"
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', str(save_dir))
        log_message(job_id, f"ZIP created: {zip_path.name}", "success")
        
        meta['state'] = 'completed'
        log_message(job_id, f"Scraping completed! Downloaded {meta['stats']['downloaded']} images", "success")
        
    except Exception as e:
        log_message(job_id, f"Scraping failed: {e}", "error")
        meta['state'] = 'failed'

def process_images_job(job_id, filenames, options):
    """Process images with real-time logging"""
    meta = JOBS[job_id]
    meta['state'] = 'processing'
    
    log_message(job_id, f"Starting processing of {len(filenames)} images...", "info")
    
    save_dir = Path(meta['folder'])
    processed_dir = save_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    def process_single_image(filename):
        if STOP_FLAGS.get(job_id, False):
            return None
        
        input_path = save_dir / filename
        if not input_path.exists():
            log_message(job_id, f"File not found: {filename}", "error")
            return None
        
        base_name = input_path.stem
        start_time = time.time()
        
        try:
            current_path = input_path
            
            # Remove background if requested
            if options.get('remove_bg') and REMBG_AVAILABLE:
                log_message(job_id, f"Removing background: {filename}", "info")
                bg_removed_path = processed_dir / f"{base_name}_nobg.png"
                
                with open(current_path, 'rb') as f:
                    input_data = f.read()
                output_data = remove(input_data)
                with open(bg_removed_path, 'wb') as f:
                    f.write(output_data)
                
                current_path = bg_removed_path
                log_message(job_id, f"Background removed: {filename}", "success")
            
            # Apply image enhancements
            with Image.open(current_path) as img:
                img = img.convert('RGB')
                
                # Brightness/Contrast
                if options.get('brightness', 1.0) != 1.0:
                    img = ImageEnhance.Brightness(img).enhance(float(options['brightness']))
                
                if options.get('contrast', 1.0) != 1.0:
                    img = ImageEnhance.Contrast(img).enhance(float(options['contrast']))
                
                # Upscaling
                if options.get('upscale'):
                    log_message(job_id, f"Upscaling: {filename}", "info")
                    w, h = img.size
                    img = img.resize((w*2, h*2), Image.LANCZOS)
                
                # Sharpening
                if options.get('enhance'):
                    img = img.filter(ImageFilter.SHARPEN)
                
                # Platform sizing
                platform = options.get('platform', 'none')
                if platform != 'none':
                    min_sizes = {'amazon': 1000, 'ebay': 500, 'shopify': 2048}
                    min_size = min_sizes.get(platform, 1000)
                    w, h = img.size
                    if min(w, h) < min_size:
                        scale = min_size / min(w, h)
                        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
                
                # Save with format conversion
                fmt = options.get('format', 'PNG').upper()
                ext = 'jpg' if fmt in ('JPG', 'JPEG') else fmt.lower()
                output_path = processed_dir / f"{base_name}_processed.{ext}"
                
                if fmt in ('JPG', 'JPEG'):
                    img.save(output_path, 'JPEG', quality=int(options.get('quality', 85)), optimize=True)
                elif fmt == 'WEBP':
                    img.save(output_path, 'WEBP', quality=int(options.get('quality', 85)), optimize=True)
                else:
                    img.save(output_path, 'PNG', optimize=True)
            
            elapsed = time.time() - start_time
            log_message(job_id, f"Processed {filename} -> {output_path.name} ({elapsed:.2f}s)", "success")
            
            # Update metadata
            with JOBS_LOCK:
                for img in meta['images']:
                    if img.get('filename') == filename:
                        img['processed_filename'] = output_path.name
                        break
            
            return output_path.name
            
        except Exception as e:
            log_message(job_id, f"Processing error for {filename}: {e}", "error")
            return None
    
    # Process in parallel
    max_workers = min(options.get('workers', 4), 8)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, fname) for fname in filenames]
        
        processed_count = 0
        for future in as_completed(futures):
            if STOP_FLAGS.get(job_id, False):
                break
            try:
                result = future.result()
                if result:
                    processed_count += 1
            except Exception as e:
                log_message(job_id, f"Processing thread error: {e}", "error")
    
    # Create processed ZIP
    try:
        zip_path = DOWNLOADS_DIR / f"{job_id}_processed.zip"
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', str(processed_dir))
        log_message(job_id, f"Created processed ZIP: {processed_count} images", "success")
    except Exception as e:
        log_message(job_id, f"ZIP creation failed: {e}", "error")
    
    meta['state'] = 'processed'
    log_message(job_id, "Processing completed!", "success")

# ----------------------- Flask Routes -----------------------

@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Shopify Image Scraper</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .log-entry { animation: fadeIn 0.3s ease-in; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="bg-gray-50">
    <div class="min-h-screen p-6">
        <div class="max-w-6xl mx-auto">
            <!-- Header -->
            <div class="text-center mb-8">
                <h1 class="text-4xl font-bold text-gray-900 mb-2">🕷️ Shopify Image Scraper</h1>
                <p class="text-gray-600">Fast concurrent scraping with real-time logs</p>
                <div class="text-sm text-gray-500 mt-2">
                    RemBG: {{ '✅ Available' if rembg_available else '❌ Not installed' }}
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Scraping Panel -->
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4">🚀 Start Scraping</h2>
                    
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium mb-1">Store URL</label>
                            <input id="storeUrl" type="url" placeholder="https://allbirds.com" 
                                   class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500">
                            <div id="productCount" class="text-sm text-gray-500 mt-1"></div>
                        </div>
                        
                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-sm font-medium mb-1">Max Products</label>
                                <input id="maxProducts" type="number" value="30" min="1" max="500"
                                       class="w-full px-3 py-2 border border-gray-300 rounded-md">
                            </div>
                            <div>
                                <label class="block text-sm font-medium mb-1">Workers</label>
                                <input id="maxWorkers" type="number" value="8" min="1" max="15"
                                       class="w-full px-3 py-2 border border-gray-300 rounded-md">
                            </div>
                        </div>
                        
                        <div class="flex items-center">
                            <input id="ignoreSSL" type="checkbox" class="mr-2" checked>
                            <label class="text-sm">Ignore SSL errors (recommended)</label>
                        </div>
                        
                        <div class="flex gap-3">
                            <button id="startBtn" onclick="startScrape()" 
                                    class="bg-green-600 text-white px-6 py-2 rounded-md font-medium hover:bg-green-700 transition">
                                🚀 Start Scrape
                            </button>
                            <button id="stopBtn" onclick="stopScrape()" 
                                    class="bg-red-600 text-white px-6 py-2 rounded-md font-medium hover:bg-red-700 transition hidden">
                                🛑 Stop
                            </button>
                        </div>
                    </div>

                    <!-- Progress -->
                    <div id="progressPanel" class="mt-6 hidden">
                        <div class="flex justify-between items-center mb-2">
                            <span id="statusText" class="font-medium">Starting...</span>
                            <span id="progressPercent" class="text-sm text-gray-600">0%</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                            <div id="progressBar" class="bg-gradient-to-r from-green-500 to-blue-500 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
                        </div>
                        <div id="statsText" class="text-sm text-gray-600 mt-2"></div>
                    </div>
                </div>

                <!-- Real-time Logs -->
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4">📋 Real-time Logs</h2>
                    <div id="logsContainer" class="bg-gray-900 text-green-400 p-4 rounded-md h-96 overflow-y-auto font-mono text-sm">
                        <div class="text-gray-500">Logs will appear here...</div>
                    </div>
                    <div class="mt-3 flex justify-between text-xs text-gray-500">
                        <span>Auto-scroll: <input id="autoScroll" type="checkbox" checked></span>
                        <button onclick="clearLogs()" class="text-red-600 hover:text-red-800">Clear Logs</button>
                    </div>
                </div>
            </div>

            <!-- Results Panel -->
            <div id="resultsPanel" class="mt-8 hidden">
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-xl font-semibold">📸 Downloaded Images</h2>
                        <div class="flex gap-3">
                            <a id="downloadLink" href="#" class="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700">
                                📥 Download ZIP
                            </a>
                            <a id="viewLink" href="#" class="bg-purple-600 text-white px-4 py-2 rounded-md hover:bg-purple-700">
                                👁️ View Gallery
                            </a>
                        </div>
                    </div>
                    <div id="imagePreview" class="grid grid-cols-4 md:grid-cols-8 gap-2"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
    let currentJobId = null;
    let polling = false;

    async function startScrape() {
        const url = document.getElementById('storeUrl').value.trim();
        if (!url) {
            alert('Please enter a store URL');
            return;
        }

        const formData = new FormData();
        formData.append('store_url', url);
        formData.append('max_products', document.getElementById('maxProducts').value);
        formData.append('max_workers', document.getElementById('maxWorkers').value);
        if (document.getElementById('ignoreSSL').checked) {
            formData.append('ignore_ssl', 'true');
        }

        try {
            // Update UI
            document.getElementById('startBtn').disabled = true;
            document.getElementById('startBtn').innerHTML = '⏳ Starting...';
            document.getElementById('stopBtn').classList.remove('hidden');
            document.getElementById('progressPanel').classList.remove('hidden');
            clearLogs();

            const response = await fetch('/api/start', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.error) {
                alert(result.error);
                resetUI();
                return;
            }

            currentJobId = result.job_id;
            addLog(`🎯 Started job: ${currentJobId}`, 'info');
            
            // Update download links
            document.getElementById('downloadLink').href = `/download/${currentJobId}.zip`;
            document.getElementById('viewLink').href = `/view/${currentJobId}`;
            
            startPolling();
            
        } catch (error) {
            alert('Failed to start scrape: ' + error.message);
            resetUI();
        }
    }

    async function stopScrape() {
        if (!currentJobId) return;
        
        try {
            await fetch(`/api/stop/${currentJobId}`, { method: 'POST' });
            addLog('🛑 Stop signal sent', 'warning');
            polling = false;
            setTimeout(resetUI, 1000);
        } catch (error) {
            alert('Failed to stop scrape: ' + error.message);
        }
    }

    function resetUI() {
        document.getElementById('startBtn').disabled = false;
        document.getElementById('startBtn').innerHTML = '🚀 Start Scrape';
        document.getElementById('stopBtn').classList.add('hidden');
        document.getElementById('progressPanel').classList.add('hidden');
    }

    async function startPolling() {
        polling = true;
        let lastLogCount = 0;
        
        while (polling && currentJobId) {
            try {
                const response = await fetch(`/api/status/${currentJobId}`);
                const status = await response.json();
                
                if (status.error) {
                    addLog(`❌ Error: ${status.error}`, 'error');
                    polling = false;
                    resetUI();
                    break;
                }

                // Update status
                const stateEmojis = {
                    'discovering': '🔍',
                    'scraping_images': '🕷️',
                    'downloading': '📥',
                    'processing': '⚙️',
                    'completed': '✅',
                    'failed': '❌',
                    'stopped': '🛑'
                };
                
                const stateText = stateEmojis[status.state] + ' ' + status.state.replace('_', ' ').toUpperCase();
                document.getElementById('statusText').textContent = stateText;
                
                const stats = status.stats;
                const progress = stats.images > 0 ? Math.round((stats.downloaded / stats.images) * 100) : 0;
                document.getElementById('progressPercent').textContent = progress + '%';
                document.getElementById('progressBar').style.width = progress + '%';
                
                document.getElementById('statsText').textContent = 
                    `Products: ${stats.products} | Images: ${stats.images} | Downloaded: ${stats.downloaded} | Errors: ${stats.errors}`;

                // Update logs (only new ones)
                if (status.logs.length > lastLogCount) {
                    const newLogs = status.logs.slice(0, status.logs.length - lastLogCount);
                    newLogs.reverse().forEach(log => addLog(log, 'stream'));
                    lastLogCount = status.logs.length;
                }
                
                // Handle completion
                if (['completed', 'failed', 'stopped', 'processed'].includes(status.state)) {
                    polling = false;
                    resetUI();
                    
                    if (status.state === 'completed') {
                        showResults();
                    }
                }
                
            } catch (error) {
                console.error('Polling error:', error);
                addLog('⚠️ Connection error during polling', 'warning');
            }
            
            await new Promise(resolve => setTimeout(resolve, 500)); // Fast polling for real-time feel
        }
    }

    function addLog(message, type = 'info') {
        const container = document.getElementById('logsContainer');
        const logDiv = document.createElement('div');
        logDiv.className = 'log-entry mb-1';
        
        const colors = {
            'info': 'text-blue-400',
            'success': 'text-green-400',
            'error': 'text-red-400',
            'warning': 'text-yellow-400',
            'stream': 'text-green-300'
        };
        
        logDiv.className += ' ' + (colors[type] || 'text-gray-400');
        logDiv.textContent = message;
        
        container.insertBefore(logDiv, container.firstChild);
        
        // Auto-scroll if enabled
        if (document.getElementById('autoScroll').checked) {
            container.scrollTop = 0;
        }
        
        // Limit log entries
        while (container.children.length > 200) {
            container.removeChild(container.lastChild);
        }
    }

    function clearLogs() {
        document.getElementById('logsContainer').innerHTML = '<div class="text-gray-500">Logs cleared...</div>';
    }

    function showResults() {
        document.getElementById('resultsPanel').classList.remove('hidden');
        addLog('✅ Scraping completed! Results panel shown', 'success');
    }

    // Auto-estimate products on URL change
    document.getElementById('storeUrl').addEventListener('blur', async function() {
        const url = this.value.trim();
        if (!url) return;
        
        document.getElementById('productCount').textContent = '🔍 Estimating products...';
        
        try {
            const response = await fetch(`/api/estimate?url=${encodeURIComponent(url)}`);
            const result = await response.json();
            
            if (result.error) {
                document.getElementById('productCount').textContent = '❌ Error: ' + result.error;
            } else {
                document.getElementById('productCount').textContent = `📊 ~${result.total} products available`;
                document.getElementById('maxProducts').max = result.total;
            }
        } catch (error) {
            document.getElementById('productCount').textContent = '⚠️ Estimation failed';
        }
    });
    </script>
</body>
</html>
    """, rembg_available=REMBG_AVAILABLE)

@app.route('/api/start', methods=['POST'])
def api_start():
    """Start scraping job"""
    store_url = request.form.get('store_url', '').strip()
    if not store_url:
        return jsonify({'error': 'Store URL required'}), 400
    
    if not store_url.startswith('http'):
        store_url = 'https://' + store_url
    
    # Create job
    job_id = uuid.uuid4().hex[:12]
    job_folder = DOWNLOADS_DIR / job_id
    job_folder.mkdir(exist_ok=True)
    
    options = {
        'max_workers': min(int(request.form.get('max_workers', 8)), 15),
        'ignore_ssl': request.form.get('ignore_ssl') == 'true',
        'max_products': int(request.form.get('max_products', 30))
    }
    
    meta = {
        'job_id': job_id,
        'store_url': store_url,
        'folder': str(job_folder),
        'options': options,
        'state': 'pending',
        'stats': {'products': 0, 'images': 0, 'downloaded': 0, 'errors': 0},
        'logs': [],
        'images': [],
        'avg_time': None,
        'created': time.time()
    }
    
    with JOBS_LOCK:
        JOBS[job_id] = meta
        STOP_FLAGS[job_id] = False
    
    # Start scraping in background
    threading.Thread(target=lambda: run_scraping_job(job_id), daemon=True).start()
    
    return jsonify({'job_id': job_id})

@app.route('/api/stop/<job_id>', methods=['POST'])
def api_stop(job_id):
    """Stop scraping job"""
    if job_id not in JOBS:
        return jsonify({'error': 'Job not found'}), 404
    
    STOP_FLAGS[job_id] = True
    log_message(job_id, "Stop signal received", "warning")
    
    return jsonify({'status': 'stopping'})

@app.route('/api/status/<job_id>')
def api_status(job_id):
    """Get job status with logs"""
    with JOBS_LOCK:
        meta = JOBS.get(job_id)
        if not meta:
            return jsonify({'error': 'Job not found'}), 404
        
        return jsonify({
            'state': meta['state'],
            'stats': meta['stats'],
            'logs': meta['logs'][:50],  # Last 50 log entries
            'avg_time': meta['avg_time']
        })

@app.route('/api/estimate')
def api_estimate():
    """Estimate total products for a store"""
    store_url = request.args.get('url', '').strip()
    if not store_url:
        return jsonify({'error': 'URL required'}), 400
    
    if not store_url.startswith('http'):
        store_url = 'https://' + store_url
    
    try:
        product_urls, error = fetch_product_urls(store_url)
        if error:
            return jsonify({'error': error}), 500
        return jsonify({'total': len(product_urls)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/view/<job_id>')
def view_results(job_id):
    """View detailed results with processing options"""
    if job_id not in JOBS:
        return "Job not found", 404
    
    meta = JOBS[job_id]
    page = int(request.args.get('page', 1))
    per_page = 24
    
    images = [img for img in meta['images'] if img.get('filename')]
    total_pages = max(1, (len(images) + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))
    
    start = (page - 1) * per_page
    end = start + per_page
    
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gallery - Shopify Scraper</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gray-50">
    <div class="min-h-screen p-6">
        <div class="max-w-7xl mx-auto">
            <!-- Header -->
            <div class="flex justify-between items-center mb-8">
                <div>
                    <h1 class="text-3xl font-bold text-gray-900">Image Gallery</h1>
                    <p class="text-gray-600">Job: {{ job_id }} | {{ meta.store_url }}</p>
                    <div class="text-sm text-gray-500 mt-1">
                        Downloaded: {{ meta.stats.downloaded }}/{{ meta.stats.images }} images
                        {% if meta.avg_time %} | Avg: {{ "%.2f"|format(meta.avg_time) }}s{% endif %}
                    </div>
                </div>
                <div class="flex gap-3">
                    <a href="/" class="bg-gray-600 text-white px-4 py-2 rounded-md">← Back to Scraper</a>
                    <a href="/download/{{ job_id }}.zip" class="bg-blue-600 text-white px-4 py-2 rounded-md">Download ZIP</a>
                </div>
            </div>

            <!-- Processing Panel -->
            {% if rembg_available %}
            <div class="bg-white rounded-lg shadow p-6 mb-8">
                <h2 class="text-xl font-semibold mb-4">Image Processing</h2>
                
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
                    <div>
                        <label class="block text-sm font-medium mb-1">Format</label>
                        <select id="format" class="w-full px-3 py-2 border rounded-md">
                            <option value="PNG">PNG</option>
                            <option value="JPG">JPG</option>
                            <option value="WEBP">WEBP</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">Quality</label>
                        <input id="quality" type="range" min="50" max="95" value="85" class="w-full">
                        <div class="text-xs text-center">85%</div>
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">Platform</label>
                        <select id="platform" class="w-full px-3 py-2 border rounded-md">
                            <option value="none">Custom</option>
                            <option value="amazon">Amazon</option>
                            <option value="ebay">eBay</option>
                            <option value="shopify">Shopify</option>
                        </select>
                    </div>
                    <div class="space-y-2">
                        <label class="flex items-center text-sm">
                            <input id="removeBg" type="checkbox" class="mr-2" checked>
                            Remove Background
                        </label>
                        <label class="flex items-center text-sm">
                            <input id="upscale" type="checkbox" class="mr-2">
                            Upscale 2x
                        </label>
                        <label class="flex items-center text-sm">
                            <input id="enhance" type="checkbox" class="mr-2">
                            Sharpen
                        </label>
                    </div>
                </div>
                
                <div class="flex gap-3 items-center">
                    <button onclick="processSelected()" class="bg-indigo-600 text-white px-6 py-2 rounded-md hover:bg-indigo-700">
                        Process Selected
                    </button>
                    <button onclick="selectAll()" class="bg-gray-600 text-white px-4 py-2 rounded-md">
                        Select All
                    </button>
                    <div id="processStatus" class="text-sm text-gray-600"></div>
                </div>
            </div>
            {% endif %}

            <!-- Image Grid -->
            <div class="bg-white rounded-lg shadow p-6">
                <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
                    {% for img in page_images %}
                    <div class="border rounded-lg overflow-hidden bg-gray-50 relative group">
                        <div class="absolute top-2 left-2 z-10">
                            <input type="checkbox" class="image-select w-4 h-4" data-filename="{{ img.filename }}">
                        </div>
                        
                        {% if img.filename %}
                        <img src="/images/{{ job_id }}/{{ img.filename }}" 
                             class="w-full h-32 object-contain bg-white cursor-pointer"
                             onclick="openImageModal('{{ img.url }}', '{{ img.filename }}')"
                             onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                        <div class="w-full h-32 flex items-center justify-center text-gray-400 hidden">
                            Failed to Load
                        </div>
                        {% else %}
                        <div class="w-full h-32 flex items-center justify-center text-red-400">
                            Download Failed
                        </div>
                        {% endif %}
                        
                        <div class="p-2 text-xs bg-white">
                            <div class="text-blue-600 truncate">
                                <a href="{{ img.url }}" target="_blank" class="hover:underline">Source</a>
                            </div>
                            {% if img.elapsed %}
                            <div class="text-gray-500">{{ "%.2f"|format(img.elapsed) }}s</div>
                            {% endif %}
                            {% if img.error %}
                            <div class="text-red-500 truncate">{{ img.error }}</div>
                            {% endif %}
                            {% if img.processed_filename %}
                            <div class="text-green-600 text-xs">✓ Processed</div>
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                </div>

                <!-- Pagination -->
                {% if total_pages > 1 %}
                <div class="flex justify-center items-center gap-4">
                    {% if page > 1 %}
                    <a href="/view/{{ job_id }}?page={{ page - 1 }}" 
                       class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">← Previous</a>
                    {% endif %}
                    
                    <span class="text-gray-600">Page {{ page }} of {{ total_pages }}</span>
                    
                    {% if page < total_pages %}
                    <a href="/view/{{ job_id }}?page={{ page + 1 }}" 
                       class="px-4 py-2 bg-gray-200 rounded hover:bg-gray-300">Next →</a>
                    {% endif %}
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <!-- Image Modal -->
    <div id="imageModal" class="fixed inset-0 bg-black bg-opacity-75 hidden z-50" onclick="closeImageModal()">
        <div class="flex items-center justify-center h-full p-4">
            <div class="bg-white rounded-lg max-w-4xl max-h-full overflow-auto">
                <div class="p-4">
                    <div class="flex justify-between items-center mb-4">
                        <h3 id="modalTitle" class="font-semibold"></h3>
                        <button onclick="closeImageModal()" class="text-gray-500 hover:text-gray-700 text-xl">×</button>
                    </div>
                    <img id="modalImage" src="" class="max-w-full h-auto">
                    <div class="mt-4">
                        <a id="modalSource" href="" target="_blank" class="text-blue-600 hover:underline">View Original Source</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
    function selectAll() {
        const checkboxes = document.querySelectorAll('.image-select');
        const allChecked = Array.from(checkboxes).every(cb => cb.checked);
        checkboxes.forEach(cb => cb.checked = !allChecked);
    }

    function openImageModal(sourceUrl, filename) {
        document.getElementById('modalTitle').textContent = filename;
        document.getElementById('modalImage').src = `/images/{{ job_id }}/${filename}`;
        document.getElementById('modalSource').href = sourceUrl;
        document.getElementById('imageModal').classList.remove('hidden');
    }

    function closeImageModal() {
        document.getElementById('imageModal').classList.add('hidden');
    }

    async function processSelected() {
        const selected = Array.from(document.querySelectorAll('.image-select:checked'))
                              .map(cb => cb.dataset.filename)
                              .filter(Boolean);
        
        if (selected.length === 0) {
            alert('Please select at least one image to process');
            return;
        }

        const options = {
            remove_bg: document.getElementById('removeBg').checked,
            upscale: document.getElementById('upscale').checked,
            enhance: document.getElementById('enhance').checked,
            format: document.getElementById('format').value,
            quality: parseInt(document.getElementById('quality').value),
            platform: document.getElementById('platform').value,
            brightness: 1.0,
            contrast: 1.0,
            workers: 4
        };

        try {
            document.getElementById('processStatus').textContent = 'Starting processing...';
            
            const response = await fetch('/api/process/{{ job_id }}', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filenames: selected, options })
            });

            const result = await response.json();
            
            if (result.error) {
                alert(result.error);
                return;
            }

            document.getElementById('processStatus').textContent = 'Processing in progress...';
            pollProcessing();
            
        } catch (error) {
            alert('Processing failed: ' + error.message);
            document.getElementById('processStatus').textContent = '';
        }
    }

    async function pollProcessing() {
        while (true) {
            try {
                const response = await fetch('/api/status/{{ job_id }}');
                const status = await response.json();
                
                if (status.state === 'processed') {
                    document.getElementById('processStatus').textContent = '✅ Processing completed!';
                    setTimeout(() => window.location.reload(), 2000);
                    break;
                } else if (status.state === 'failed') {
                    document.getElementById('processStatus').textContent = '❌ Processing failed';
                    break;
                }
                
                await new Promise(resolve => setTimeout(resolve, 1000));
            } catch (error) {
                break;
            }
        }
    }

    // Quality slider update
    document.getElementById('quality').addEventListener('input', function() {
        this.nextElementSibling.textContent = this.value + '%';
    });
    </script>
</body>
</html>
    """, 
    job_id=job_id, 
    meta=meta, 
    page_images=images[start:end],
    page=page,
    total_pages=total_pages,
    rembg_available=REMBG_AVAILABLE
    )

@app.route('/api/process/<job_id>', methods=['POST'])
def api_process(job_id):
    """Process selected images"""
    if job_id not in JOBS:
        return jsonify({'error': 'Job not found'}), 404
    
    data = request.get_json()
    if not data or not data.get('filenames'):
        return jsonify({'error': 'No filenames provided'}), 400
    
    threading.Thread(
        target=lambda: process_images_job(job_id, data['filenames'], data.get('options', {})),
        daemon=True
    ).start()
    
    return jsonify({'status': 'Processing started'})

@app.route('/images/<job_id>/<filename>')
def serve_image(job_id, filename):
    """Serve images from job folder"""
    if job_id not in JOBS:
        return "Job not found", 404
    
    job_folder = Path(JOBS[job_id]['folder'])
    
    for folder in [job_folder, job_folder / 'processed']:
        file_path = folder / filename
        if file_path.exists():
            return send_from_directory(str(folder), filename)
    
    return "Image not found", 404

@app.route('/download/<filename>')
def download_file(filename):
    """Download ZIP files"""
    file_path = DOWNLOADS_DIR / filename
    if not file_path.exists():
        return "File not found", 404
    
    return send_from_directory(str(DOWNLOADS_DIR), filename, as_attachment=True)

# ----------------------- Background Cleanup -----------------------

def cleanup_old_jobs():
    """Clean up expired jobs"""
    while True:
        try:
            with JOBS_LOCK:
                current_time = time.time()
                expired = [jid for jid, meta in JOBS.items() 
                          if current_time - meta['created'] > 86400]  # 24 hours
                
                for job_id in expired:
                    job_folder = Path(JOBS[job_id]['folder'])
                    if job_folder.exists():
                        shutil.rmtree(job_folder, ignore_errors=True)
                    
                    for suffix in ['', '_processed']:
                        zip_file = DOWNLOADS_DIR / f"{job_id}{suffix}.zip"
                        if zip_file.exists():
                            zip_file.unlink()
                    
                    del JOBS[job_id]
                    if job_id in STOP_FLAGS:
                        del STOP_FLAGS[job_id]
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        time.sleep(3600)

# ----------------------- Main Application -----------------------

if __name__ == '__main__':
    print("🚀 Starting Optimized Shopify Image Scraper...")
    print(f"📁 Downloads Directory: {DOWNLOADS_DIR}")
    print(f"🔧 RemBG Available: {REMBG_AVAILABLE}")
    
    # Start cleanup thread
    threading.Thread(target=cleanup_old_jobs, daemon=True).start()
    
    # Run Flask app
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
