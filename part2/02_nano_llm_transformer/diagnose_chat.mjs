import { chromium } from 'playwright';

const PROMPTS = [
  'What is the capital of France?',
  'Give three tips for staying healthy.',
  'Write a short poem about the ocean.',
  'What color is the sky?',
  'Count from one to five.',
  'Tell me about dogs.',
];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 900, height: 1200 } });
const errors = [];
const failed = [];
page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });
page.on('pageerror', (err) => errors.push('pageerror: ' + err.message));
page.on('requestfailed', (req) => failed.push(req.url()));

await page.goto('http://localhost:5175', { waitUntil: 'networkidle', timeout: 20000 });
await page.waitForSelector('text=NanoLlama');

const results = [];
for (const prompt of PROMPTS) {
  await page.fill('input[placeholder*="Ask NanoLlama"]', prompt);
  await page.click('button:has-text("Send")');
  await page.waitForSelector('text=tok/s', { timeout: 20000 });
  await page.waitForTimeout(300);
  const bubbles = await page.locator('.whitespace-pre-wrap').allTextContents();
  const reply = bubbles[bubbles.length - 1];
  results.push({ prompt, reply });
  console.log(`\nPROMPT: ${prompt}\n  -> ${reply}`);
}

await page.screenshot({ path: 'docs/screenshots/chat_diagnostic.png', fullPage: true });

console.log('\nConsole/page errors:', errors.length === 0 ? 'none' : JSON.stringify(errors));
console.log('Failed requests:', failed.length === 0 ? 'none' : JSON.stringify(failed));

await browser.close();
