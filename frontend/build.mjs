import {build} from 'esbuild';
import {mkdir, copyFile} from 'node:fs/promises';
await mkdir('dist', {recursive: true});
await build({entryPoints: ['src/main.mjs'], bundle: true, minify: true,
  outfile: 'dist/app.js', define: {'process.env.NODE_ENV': '"production"'}});
await copyFile('index.html', 'dist/index.html');
await copyFile('src/style.css', 'dist/style.css');
