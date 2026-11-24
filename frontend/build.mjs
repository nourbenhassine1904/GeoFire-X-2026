import {build} from 'esbuild';
import {mkdir, copyFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {dirname,resolve} from 'node:path';
const directory=dirname(fileURLToPath(import.meta.url));
process.chdir(directory);
await mkdir('dist', {recursive: true});
await build({absWorkingDir:directory,tsconfigRaw:{},entryPoints: [resolve(directory,'src/main.mjs')], bundle: true, minify: true,
  outfile: 'dist/app.js', define: {'process.env.NODE_ENV': '"production"'}});
await copyFile('index.html', 'dist/index.html');
await copyFile('src/style.css', 'dist/style.css');
await copyFile('src/reactbits/LICENSE.md', 'dist/react-bits-LICENSE.txt');
