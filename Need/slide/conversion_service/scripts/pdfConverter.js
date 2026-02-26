const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { PDFDocument } = require('pdf-lib');
const winston = require('winston');

// Configure simple logger for this script
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.json(),
  transports: [
    new winston.transports.Console({
      format: winston.format.simple(),
    })
  ],
});

/**
 * Sanitize HTML to remove potentially dangerous local file and data URLs
 */
function sanitizeHtml(htmlContent) {
  // Remove file:// and data:// URLs from src, href, and other attributes
  return htmlContent
    .replace(/\bfile:\/\/[^\s"'>]*/gi, '#')
    .replace(/\bdata:[^\s"'>]*/gi, '#');
}

/**
 * Create temporary HTML file from content
 */
function createTempHtmlFile(htmlContent, slideIndex) {
  const tempDir = path.join(os.tmpdir(), 'slide-pdf-generator');

  if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir, { recursive: true });
  }

  const filePath = path.join(tempDir, `slide-${slideIndex}-${Date.now()}.html`);
  fs.writeFileSync(filePath, htmlContent, 'utf8');

  return filePath;
}

/**
 * Delete temporary HTML file
 */
function deleteTempHtmlFile(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch (err) {
    logger.warn(`Could not delete temp file: ${filePath}`);
  }
}

/**
 * Convert slides array to PDF Buffer
 * @param {Array} slides - Array of slide objects with html_content property
 * @param {Object} options - Configuration options
 * @returns {Promise<Buffer>} PDF buffer
 */
async function convertToPdf(slides, options = {}) {
  const {
    slideTimeout = 120000,
  } = options;

  if (!slides || !Array.isArray(slides) || slides.length === 0) {
    throw new Error('Slides must be a non-empty array');
  }

  let browser = null;
  let page = null;
  const tempFiles = [];

  try {
    logger.info(`Starting PDF conversion for ${slides.length} slides`);

    // Launch Puppeteer browser
    browser = await puppeteer.launch({
      executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
      headless: 'new',
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
      ],
    });

    // Create new page with fixed viewport
    page = await browser.newPage();
    await page.setViewport({
      width: 1280,
      height: 720,
      deviceScaleFactor: 2,
    });

    // Initialize final PDF document
    const finalPdf = await PDFDocument.create();

    // Process each slide
    for (let i = 0; i < slides.length; i++) {
      const slide = slides[i];

      if (!slide) {
        logger.warn(`Slide ${i + 1} is null or undefined`);
        continue;
      }

      const slideHtml = slide.html_content;

      if (!slideHtml) {
        logger.warn(`Slide ${i + 1} has no html_content`);
        continue;
      }

      try {
        // Sanitize HTML content to remove dangerous file:// and data:// URLs
        const sanitizedHtml = sanitizeHtml(slideHtml);

        logger.info(`Processing slide ${i + 1}/${slides.length}`);

        // Load HTML directly using page.setContent for safety (avoids file:// access)
        await page.setContent(sanitizedHtml, {
          waitUntil: 'domcontentloaded',
          timeout: slideTimeout,
        });

        // Generate PDF for this slide
        const pdfBytes = await page.pdf({
          width: '1280px',
          height: '720px',
          printBackground: true,
          pageRanges: '1',
          margin: {
            top: '0px',
            right: '0px',
            bottom: '0px',
            left: '0px',
          },
        });

        // Load and merge into final PDF
        const slidePdf = await PDFDocument.load(pdfBytes);
        const [slidePage] = await finalPdf.copyPages(slidePdf, [0]);
        finalPdf.addPage(slidePage);

      } catch (err) {
        logger.error(`Error processing slide ${i + 1}: ${err.message}`);
        throw err; // Fail hard for now, or we could skip
      }
    }

    // Check if any pages were added to the PDF
    const pageCount = finalPdf.getPageCount();
    if (pageCount === 0) {
      throw new Error('No slides with html_content; no PDF generated');
    }

    // Save final merged PDF to buffer
    const finalPdfBytes = await finalPdf.save();
    return Buffer.from(finalPdfBytes);

  } catch (err) {
    logger.error(`Fatal PDF conversion error: ${err.message}`);
    throw err;
  } finally {
    if (page) await page.close().catch(() => { });
    if (browser) await browser.close().catch(() => { });

    // Delete temporary files
    tempFiles.forEach(filePath => {
      deleteTempHtmlFile(filePath);
    });
  }
}

module.exports = { convertToPdf };
