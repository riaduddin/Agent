const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const Automizer = require('pptx-automizer').default;    
const winston = require('winston');

// Configure simple logger
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.json(),
    transports: [
        new winston.transports.Console({
            format: winston.format.simple(),
            stderrLevels: ['error'],
        })
    ],
});

// /**
//  * Wait for images and styles to load in the page with timeout
//  */
// async function waitForResources(page, maxWaitTime = 10000) {
//     try {
//         await page.evaluate((timeout) => {
//             return new Promise((resolve) => {
//                 const startTime = Date.now();

//                 const checkImages = () => {
//                     const images = Array.from(document.images);
//                     return images.length === 0 || images.every(img => img.complete && img.naturalHeight !== 0);
//                 };

//                 const checkStylesheets = () => {
//                     const stylesheets = Array.from(document.styleSheets);
//                     return stylesheets.every(sheet => {
//                         try {
//                             return sheet.cssRules || sheet.sheet === null;
//                         } catch (e) {
//                             return true;
//                         }
//                     });
//                 };

//                 const interval = setInterval(() => {
//                     const elapsed = Date.now() - startTime;
//                     const imagesReady = checkImages();
//                     const stylesheetsReady = checkStylesheets();

//                     if (imagesReady && stylesheetsReady) {
//                         clearInterval(interval);
//                         resolve();
//                     } else if (elapsed >= timeout) {
//                         clearInterval(interval);
//                         resolve();
//                     }
//                 }, 500);
//             });
//         }, maxWaitTime);
//     } catch (err) {
//         logger.warn(`Resource check error: ${err.message}`);
//     }
// }

// /**
//  * Convert a single slide to PPTX blob with a fresh browser page
//  */
// async function renderSlideToBlob(browser, slideContent, slideIndex) {
//     const page = await browser.newPage();

//     try {
//         // Set viewport to match slide dimensions
//         await page.setViewport({ width: 1280, height: 720, deviceScaleFactor: 1 });

//         // Sanitize slide content before embedding in HTML template
//         // const sanitizedContent = sanitizeHtml(slideContent);

//         // Create isolated HTML for this single slide with restrictive Content Security Policy
//         const slideHtml = `<!DOCTYPE html>
//       <html>
//       <head>
//         <meta charset="UTF-8">
//         <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: https:; font-src 'self' data:;">
//         <style>
//           * { margin: 0; padding: 0; box-sizing: border-box; }
//           body { width: 1280px; height: 720px; overflow: hidden; }
//           .slide-wrapper { width: 100%; height: 100%; }
//         </style>
//       </head>
//       <body>
//         <div class="slide-wrapper" id="slide-content">${slideContent}</div>
//       </body>
//       </html>
//     `;

//         // Load the slide into the page
//         await page.setContent(slideHtml, { waitUntil: 'domcontentloaded', timeout: 120000 });

//         // Wait for resources to load
//         await waitForResources(page);

//         // Check if slide has a chart/canvas element
//         const hasChart = await page.evaluate(() => {
//             return document.querySelector('canvas') !== null;
//         });

//         if (hasChart) {
//             logger.info(`Slide ${slideIndex + 1}: Chart detected. Injecting Chart.js...`);
//             await page.evaluate(() => {
//                 return new Promise((resolve) => {
//                     if (window.Chart) {
//                         resolve();
//                         return;
//                     }
//                     const script = document.createElement('script');
//                     script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
//                     script.onload = () => resolve();
//                     script.onerror = () => resolve();
//                     setTimeout(() => resolve(), 5000);
//                     document.head.appendChild(script);
//                 });
//             });

//             // Wait for Chart.js to fully initialize
//             await new Promise(resolve => setTimeout(resolve, 1500));
//         }

//         // Inject dom-to-pptx library. Note: We resolve this relative to THIS script location to bypass package export restrictions.
//         // In the docker container and local, node_modules will be in the parent of scripts/
//         const bundlePath = path.resolve(__dirname, '../node_modules/dom-to-pptx/dist/dom-to-pptx.bundle.js');
//         if (!fs.existsSync(bundlePath)) {
//             throw new Error(`dom-to-pptx bundle not found at ${bundlePath}`);
//         }
//         await page.addScriptTag({ path: bundlePath });

//         // Export this slide to PPTX with timeout protection
//         const pptxBase64 = await Promise.race([
//             page.evaluate(async () => {
//                 if (typeof domToPptx === 'undefined') {
//                     throw new Error('domToPptx library not found');
//                 }

//                 const blob = await domToPptx.exportToPptx(['#slide-content'], {
//                     fileName: `slide-${Date.now()}.pptx`,
//                     skipDownload: true
//                 });

//                 return new Promise((resolve, reject) => {
//                     const reader = new FileReader();
//                     reader.onloadend = () => {
//                         const base64 = reader.result.split(',')[1];
//                         resolve(base64);
//                     };
//                     reader.onerror = (e) => reject(new Error('FileReader error: ' + e.target.error));
//                     reader.readAsDataURL(blob);
//                 });
//             }),
//             new Promise((_, reject) =>
//                 setTimeout(() => reject(new Error('PPTX export timeout after 30 seconds')), 30000)
//             )
//         ]);

//         return Buffer.from(pptxBase64, 'base64');

//     } finally {
//         await page.close();
//     }
// }

// /**
//  * Merge multiple PPTX files into one using pptx-automizer
//  */
// async function mergePptxFiles(pptxFiles) {
//     logger.info(`Merging ${pptxFiles.length} PPTX files...`);

//     const automizer = new Automizer({
//         templateDir: path.dirname(pptxFiles[0]),
//         outputDir: path.dirname(pptxFiles[0]),
//         useCreationIds: false,
//         autoImportSlideMasters: false,
//         removeExistingSlides: false,
//         cleanup: false
//     });

//     // Load first file as root
//     let pres = automizer.loadRoot(path.basename(pptxFiles[0]));

//     // Load remaining files as templates
//     if (pptxFiles.length > 1) {
//         for (let i = 1; i < pptxFiles.length; i++) {
//             pres.load(path.basename(pptxFiles[i]), `template-${i}`);
//         }
//     }

//     // Add slides from each template
//     for (let i = 1; i < pptxFiles.length; i++) {
//         const slideNumbers = await pres.getTemplate(`template-${i}`).getAllSlideNumbers();
//         for (const slideNum of slideNumbers) {
//             pres.addSlide(`template-${i}`, slideNum);
//         }
//     }

//     // Generate final merged PPTX
//     const jszip = await pres.getJSZip();
//     const mergedBuffer = await jszip.generateAsync({ type: 'nodebuffer' });

//     return mergedBuffer;
// }

// /**
//  * Convert slides array to PPTX Buffer
//  */
// async function convertToPptx(slides, options = {}) {
//     const { jobId = 'unknown' } = options;

//     if (!slides || !Array.isArray(slides) || slides.length === 0) {
//         throw new Error('Slides must be a non-empty array');
//     }

//     let browser = null;
//     const tempFiles = [];
//     const pptxBuffers = [];
//     let tempDir = null;  // Track tempDir for cleanup in finally block

//     try {
//         logger.info(`Starting PPTX conversion for ${slides.length} slides`);

//         // Determine if web security should be disabled (only for controlled environments)
//         const allowInsecureBrowser = process.env.ALLOW_INSECURE_BROWSER === 'true';
//         if (allowInsecureBrowser) {
//             logger.warn('WARNING: Running with disabled web security - only use in controlled Docker/dev environments');
//         }

//         // Launch Puppeteer browser
//         browser = await puppeteer.launch({
//             headless: 'new',
//             args: [
//                 // Required for Docker and containerized environments to run Chromium without sandbox
//                 '--no-sandbox',
//                 '--disable-setuid-sandbox',
//                 // Only disable web security in controlled environments (gated by ALLOW_INSECURE_BROWSER env var)
//                 // This is needed for some HTML rendering scenarios but should not be used in production
//                 ...(allowInsecureBrowser ? ['--disable-web-security'] : []),
//                 // Disable process isolation (required for rendering complex slides)
//                 '--disable-features=IsolateOrigins,site-per-process'
//             ],
//         });

//         // Process each slide individually
//         for (let i = 0; i < slides.length; i++) {
//             const slide = slides[i];
//             const slideHtml = slide.html_content;

//             if (!slideHtml) {
//                 logger.warn(`Slide ${i + 1} has no html_content`);
//                 continue;
//             }

//             try {
//                 const dom = new JSDOM(slideHtml);
//                 const doc = dom.window.document;

//                 // Extract styles and body content
//                 const styles = Array.from(doc.querySelectorAll('style')).map(s => s.textContent).join('\n');
//                 const bodyContent = doc.body.innerHTML;

//                 // Combine with scoped styles
//                 const slideHtmlContent = `
//           <style>${styles}</style>
//           ${bodyContent}
//         `;

//                 logger.info(`Processing slide ${i + 1}/${slides.length}`);

//                 // Render slide to blob
//                 const buffer = await renderSlideToBlob(browser, slideHtmlContent, i);
//                 pptxBuffers.push({ slideNumber: i + 1, buffer, success: true });

//             } catch (err) {
//                 logger.error(`Error processing slide ${i + 1}: ${err.message}`);
//                 pptxBuffers.push({ slideNumber: i + 1, buffer: null, success: false, error: err.message });
//             }
//         }

//         const successCount = pptxBuffers.filter(s => s.success).length;
//         if (successCount === 0) {
//             throw new Error('No slides were successfully exported');
//         }

//         // Save each successful slide as a separate PPTX file in temp
//         tempDir = path.join(require('os').tmpdir(), `pptx-conversion-${jobId}-${Date.now()}`);
//         if (!fs.existsSync(tempDir)) {
//             fs.mkdirSync(tempDir, { recursive: true });
//         }

//         const savedFiles = [];
//         for (let i = 0; i < pptxBuffers.length; i++) {
//             if (pptxBuffers[i].success) {
//                 const outputFileName = `output-slide-${pptxBuffers[i].slideNumber}.pptx`;
//                 const filePath = path.join(tempDir, outputFileName);
//                 fs.writeFileSync(filePath, pptxBuffers[i].buffer);
//                 savedFiles.push(filePath);
//                 tempFiles.push(filePath);
//             }
//         }

//         // Merge all PPTX files
//         let mergedBuffer;
//         if (savedFiles.length > 1) {
//             mergedBuffer = await mergePptxFiles(savedFiles);
//         } else {
//             mergedBuffer = pptxBuffers.find(b => b.success).buffer;
//         }

//         logger.info(`PPTX conversion completed: ${successCount} slides`);
//         return mergedBuffer;

//     } catch (err) {
//         logger.error(`Fatal PPTX conversion error: ${err.message}`);
//         throw err;
//     } finally {
//         if (browser) await browser.close().catch(() => { });

//         // Delete temporary files
//         tempFiles.forEach(filePath => {
//             try {
//                 if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
//             } catch (e) { }
//         });

//         // Clean up temporary directory
//         if (tempDir) {
//             try {
//                 if (fs.existsSync(tempDir)) {
//                     fs.rmSync(tempDir, { recursive: true, force: true });
//                 }
//             } catch (err) {
//                 logger.warn(`Could not remove temp directory ${tempDir}: ${err.message}`);
//             }
//         }
//     }
// }

// module.exports = { convertToPptx };

/**
 * Wait for images and styles to load in the page with timeout
 */
async function waitForResources(page, maxWaitTime = 10000) {
    try {
        await page.evaluate((timeout) => {
            return new Promise((resolve) => {
                const startTime = Date.now();

                const checkImages = () => {
                    const images = Array.from(document.images);
                    return images.length === 0 || images.every(img => img.complete && img.naturalHeight !== 0);
                };

                const checkStylesheets = () => {
                    const stylesheets = Array.from(document.styleSheets);
                    return stylesheets.every(sheet => {
                        try {
                            return sheet.cssRules || sheet.sheet === null;
                        } catch (e) {
                            return true;
                        }
                    });
                };

                const interval = setInterval(() => {
                    const elapsed = Date.now() - startTime;
                    const imagesReady = checkImages();
                    const stylesheetsReady = checkStylesheets();

                    if (imagesReady && stylesheetsReady) {
                        clearInterval(interval);
                        resolve();
                    } else if (elapsed >= timeout) {
                        clearInterval(interval);
                        resolve();
                    }
                }, 500);
            });
        }, maxWaitTime);
    } catch (err) {
        logger.warn(`Resource check error: ${err.message}`);
    }
}

/**
 * Convert a single slide to PPTX blob with a fresh browser page
 */
async function renderSlideToBlob(browser, slideContent, slideIndex, jobId) {
    const page = await browser.newPage();

    try {
        // Set viewport to match slide dimensions
        await page.setViewport({ width: 1280, height: 720, deviceScaleFactor: 1 });

        // Create isolated HTML for this single slide
        const slideHtml = `<!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8">
        <style>
          * { margin: 0; padding: 0; box-sizing: border-box; }
          body { width: 1280px; height: 720px; overflow: hidden; }
          .slide-wrapper { width: 100%; height: 100%; }
        </style>
      </head>
      <body>
        <div class="slide-wrapper" id="slide-content">${slideContent}</div>
      </body>
      </html>
    `;

        // Load the slide into the page
        await page.setContent(slideHtml, { waitUntil: 'domcontentloaded', timeout: 120000 });

        // Wait for resources to load
        await waitForResources(page);

        // Check if slide has a chart/canvas element
        const hasChart = await page.evaluate(() => {
            return document.querySelector('canvas') !== null;
        });

        if (hasChart) {
            logger.info(`Slide ${slideIndex + 1}: Chart detected. Injecting Chart.js...`, { jobId });
            await page.evaluate(() => {
                return new Promise((resolve) => {
                    if (window.Chart) {
                        resolve();
                        return;
                    }
                    const script = document.createElement('script');
                    script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
                    script.onload = () => resolve();
                    script.onerror = () => resolve();
                    setTimeout(() => resolve(), 5000);
                    document.head.appendChild(script);
                });
            });

            // Wait for Chart.js to fully initialize
            await new Promise(resolve => setTimeout(resolve, 1500));
        }

        // Inject dom-to-pptx library
        const bundlePath = path.resolve(__dirname, '../node_modules/dom-to-pptx/dist/dom-to-pptx.bundle.js');
        await page.addScriptTag({ path: bundlePath });

        // Export this slide to PPTX with timeout protection
        const pptxBase64 = await Promise.race([
            page.evaluate(async () => {
                if (typeof domToPptx === 'undefined') {
                    throw new Error('domToPptx library not found');
                }

                const blob = await domToPptx.exportToPptx(['#slide-content'], {
                    fileName: `slide-${Date.now()}.pptx`,
                    skipDownload: true
                });

                return new Promise((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64 = reader.result.split(',')[1];
                        resolve(base64);
                    };
                    reader.onerror = (e) => reject(new Error('FileReader error: ' + e.target.error));
                    reader.readAsDataURL(blob);
                });
            }),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error('PPTX export timeout after 30 seconds')), 30000)
            )
        ]);

        return Buffer.from(pptxBase64, 'base64');

    } finally {
        await page.close();
    }
}

/**
 * Merge multiple PPTX files into one using pptx-automizer
 */
async function mergePptxFiles(pptxFiles, jobId) {
    logger.info(`Merging ${pptxFiles.length} PPTX files...`, { jobId });

    const automizer = new Automizer({
        templateDir: path.dirname(pptxFiles[0]),
        outputDir: path.dirname(pptxFiles[0]),
        useCreationIds: false,
        autoImportSlideMasters: false,
        removeExistingSlides: false,
        cleanup: false
    });

    // Load first file as root
    let pres = automizer.loadRoot(path.basename(pptxFiles[0]));

    // Load remaining files as templates
    if (pptxFiles.length > 1) {
        for (let i = 1; i < pptxFiles.length; i++) {
            pres.load(path.basename(pptxFiles[i]), `template-${i}`);
        }
    }

    // Add slides from each template
    for (let i = 1; i < pptxFiles.length; i++) {
        const slideNumbers = await pres.getTemplate(`template-${i}`).getAllSlideNumbers();
        for (const slideNum of slideNumbers) {
            pres.addSlide(`template-${i}`, slideNum);
        }
    }

    // Generate final merged PPTX
    const jszip = await pres.getJSZip();
    const mergedBuffer = await jszip.generateAsync({ type: 'nodebuffer' });

    logger.info(`Merge complete: ${pptxFiles.length} files combined`, { jobId });
    return mergedBuffer;
}

/**
 * Convert slides array to PPTX Buffer
 * @param {Array} slides - Array of slide objects with html_content property
 * @param {Object} options - Configuration options
 * @param {Function} options.onProgress - Callback for progress tracking
 * @param {string} options.jobId - Job ID for logging
 * @returns {Promise<Buffer>} PPTX buffer
 */
async function convertToPptx(slides, options = {}) {
    const {
        onProgress = null,
        jobId = 'unknown',
    } = options;

    if (!slides || !Array.isArray(slides) || slides.length === 0) {
        throw new Error('Slides must be a non-empty array');
    }

    let browser = null;
    const tempFiles = [];
    const pptxBuffers = [];

    try {
        logger.info(`Starting PPTX conversion for ${slides.length} slides`, { jobId });

        // Launch Puppeteer browser
        browser = await puppeteer.launch({
            headless: 'new',
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process'
            ],
        });

        // Process each slide individually
        for (let i = 0; i < slides.length; i++) {
            const slide = slides[i];
            const slideHtml = slide.html_content;

            if (!slideHtml) {
                logger.warn(`Slide ${i + 1} has no html_content`, { jobId });
                if (onProgress) {
                    onProgress({
                        currentSlide: i + 1,
                        totalSlides: slides.length,
                        status: 'rendering',
                        message: `Skipped slide ${i + 1} (no content)`,
                    });
                }
                continue;
            }

            try {
                const dom = new JSDOM(slideHtml);
                const doc = dom.window.document;

                // Extract styles and body content
                const styles = Array.from(doc.querySelectorAll('style')).map(s => s.textContent).join('\n');
                const bodyContent = doc.body.innerHTML;

                // Combine with scoped styles
                const slideHtmlContent = `
          <style>${styles}</style>
          ${bodyContent}
        `;

                logger.info(`Processing slide ${i + 1}/${slides.length}`, { jobId });

                // Render slide to blob
                const buffer = await renderSlideToBlob(browser, slideHtmlContent, i, jobId);
                pptxBuffers.push({ slideNumber: i + 1, buffer, success: true });

                if (onProgress) {
                    onProgress({
                        currentSlide: i + 1,
                        totalSlides: slides.length,
                        status: 'rendering',
                        progress: ((i + 1) / slides.length) * 100,
                        message: `Rendered slide ${i + 1}`,
                    });
                }

            } catch (err) {
                logger.error(`Error processing slide ${i + 1}: ${err.message}`, { jobId });
                pptxBuffers.push({ slideNumber: i + 1, buffer: null, success: false, error: err.message });

                if (onProgress) {
                    onProgress({
                        currentSlide: i + 1,
                        totalSlides: slides.length,
                        status: 'error',
                        message: `Failed to render slide ${i + 1}`,
                    });
                }
            }
        }

        const successCount = pptxBuffers.filter(s => s.success).length;
        const failCount = pptxBuffers.filter(s => !s.success).length;

        logger.info(`Export summary: ${successCount} successful, ${failCount} failed`, { jobId });

        if (successCount === 0) {
            throw new Error('No slides were successfully exported');
        }

        // Save each successful slide as a separate PPTX file in temp
        const tempDir = path.join(require('os').tmpdir(), `pptx-conversion-${jobId}`);
        if (!fs.existsSync(tempDir)) {
            fs.mkdirSync(tempDir, { recursive: true });
        }

        const savedFiles = [];
        for (let i = 0; i < pptxBuffers.length; i++) {
            if (pptxBuffers[i].success) {
                const outputFileName = `output-slide-${pptxBuffers[i].slideNumber}.pptx`;
                const filePath = path.join(tempDir, outputFileName);
                fs.writeFileSync(filePath, pptxBuffers[i].buffer);
                savedFiles.push(filePath);
                tempFiles.push(filePath);
            }
        }

        // Merge all PPTX files
        let mergedBuffer;
        if (savedFiles.length > 1) {
            if (onProgress) {
                onProgress({
                    currentSlide: slides.length,
                    totalSlides: slides.length,
                    status: 'uploading',
                    message: 'Merging PPTX files...',
                });
            }
            mergedBuffer = await mergePptxFiles(savedFiles, jobId);
        } else {
            // Only one slide
            mergedBuffer = pptxBuffers.find(b => b.success).buffer;
        }

        logger.info(`PPTX conversion completed: ${successCount} slides`, { jobId });

        return mergedBuffer;

    } catch (err) {
        logger.error(`Fatal PPTX conversion error: ${err.message}`, { jobId });
        throw err;
    } finally {
        // Cleanup
        if (browser) {
            try {
                await browser.close();
            } catch (err) {
                logger.warn(`Error closing browser: ${err.message}`, { jobId });
            }
        }

        // Delete temporary files
        tempFiles.forEach(filePath => {
            try {
                if (fs.existsSync(filePath)) {
                    fs.unlinkSync(filePath);
                }
            } catch (err) {
                logger.warn(`Could not delete temp file: ${filePath}`, { jobId });
            }
        });
    }
}

module.exports = { convertToPptx };