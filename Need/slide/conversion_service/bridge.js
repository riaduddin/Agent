const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { convertToPdf } = require('./scripts/pdfConverter');
const { convertToPptx } = require('./scripts/pptxConverter');
const fs = require('fs');
const winston = require('winston');

// Logger setup
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

async function main() {
    const argv = yargs(hideBin(process.argv))
        .option('format', {
            alias: 'f',
            description: 'Output format (pdf or pptx)',
            type: 'string',
            choices: ['pdf', 'pptx'],
            demandOption: true,
        })
        .option('input', {
            alias: 'i',
            description: 'Path to input JSON file containing slides',
            type: 'string',
            demandOption: true,
        })
        .option('output', {
            alias: 'o',
            description: 'Path to output file',
            type: 'string',
            demandOption: true,
        })
        .option('jobId', {
            alias: 'j',
            description: 'Job ID for logging',
            type: 'string',
        })
        .help()
        .argv;

    try {
        const inputPath = argv.input;
        const outputPath = argv.output;
        const format = argv.format;
        const jobId = argv.jobId || 'unknown';

        if (!fs.existsSync(inputPath)) {
            throw new Error(`Input file not found: ${inputPath}`);
        }

        const inputData = fs.readFileSync(inputPath, 'utf8');
        const slides = JSON.parse(inputData);

        logger.info(`Starting conversion job ${jobId} to ${format}`);

        let buffer;
        if (format === 'pdf') {
            buffer = await convertToPdf(slides, { jobId });
        } else if (format === 'pptx') {
            buffer = await convertToPptx(slides, { jobId });
        }

        fs.writeFileSync(outputPath, buffer);
        logger.info(`Conversion successful. Output written to ${outputPath}`);

    } catch (error) {
        logger.error('Conversion failed:', error);
        process.exit(1);
    }
}

main();
