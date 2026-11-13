// Uses the SAME request routine, React view and pixel renderer as the live UI.
// This exercises HTTP -> FastAPI -> real PyTorch -> React -> canvas pixel bytes.
// Browser interactions remain a separate manual check when its host is unavailable.
import {build} from 'esbuild';
import {execFileSync} from 'node:child_process';
await build({stdin:{contents:`
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {ForecastView} from './src/EventForecast.mjs';
import {ComparisonView} from './src/PropagationComparison.mjs';
import {requestForecast,comparisonLayers,imagePixels,forecastCSV} from './src/prediction_view.mjs';
import fs from 'node:fs';
async function main(){
 // Unit coverage of the SAME export/mask/retry functions used in the browser.
 const fake={event_id:'2020/example',as_of:'2020-01-01',target_date:'2020-01-02',checkpoint_sha256:'fixture',threshold:.78,preview_stride:1,full_shape:[2,2],crs:null,transform:null,probability:[[0,.8],[null,.9]],observed_fire:[[0,0],[null,1]],predicted_fire:[[0,1],[null,1]],new_predicted_fire:[[0,1],[null,0]]};
 const fakeTruth=[[0,1],[null,1]],exported=forecastCSV(fake,fakeTruth).split('\\n');
 const invalid=exported[3].split(',');
 if(exported.length!==5||invalid[8]!==''||invalid[10]!==''||invalid[14]!=='')throw new Error('CSV invalid pixels became negative labels');
 const palette=imagePixels([[0,1,2,3,null]],'errors').pixels;
 if(palette[8]!==242||palette[9]!==82||palette[12]!==63||palette[13]!==142||palette[16]!==85)throw new Error('Error colours differ from legend');
 const realFetch=globalThis.fetch;let busyAttempts=0;
 globalThis.fetch=async url=>{
   if(url.includes('/predict?')&&busyAttempts++===0)return new Response(JSON.stringify({detail:'busy'}),{status:409});
   return new Response(JSON.stringify(url.includes('/ground-truth?')?{...fake,ground_truth:fakeTruth}:fake),{status:200});
 };
 try{const retry=await requestForecast('http://fixture/lab',fake.event_id,fake.as_of);if(busyAttempts!==2||retry.truthError||!retry.truth)throw new Error('Busy retry failed');}
 finally{globalThis.fetch=realFetch;}
 const base='http://127.0.0.1:8014';
 const status=await (await fetch(base+'/lab/status')).json();
 const events=await (await fetch(base+'/lab/events')).json();
 if(!status.ready||status.threshold!==.78||events.length!==201)throw new Error('Wrong retained model/catalogue');
 let intermediate=false;
 const response=await requestForecast(base+'/lab','2020/fire_24332880','2020-09-03',{onPrediction:r=>{if(r.target_pixels_read!==false)throw new Error('Future label reached inference');intermediate=true;}});
 if(!intermediate||response.truthError||!response.truth)throw new Error('Forecast/truth pipeline failed');
 const {result,truth}=response;
 const html=renderToStaticMarkup(React.createElement(ForecastView,{result,truth}));
 if((html.match(/<canvas /g)||[]).length!==8)throw new Error('Eight live maps required');
 const layers=comparisonLayers(result,truth),image=imagePixels(result.predicted_fire);
 const errors=imagePixels(layers.newErrors,'errors');
 if(image.pixels.length!==image.width*image.height*4||errors.pixels.length!==image.pixels.length)throw new Error('Canvas buffers differ');
 const csv=forecastCSV(result,truth);
 if(csv.split('\\n').length!==image.width*image.height+1||!csv.includes(result.checkpoint_sha256))throw new Error('Pixel CSV/provenance export differs');
 if(!html.includes('Agrandir')||!html.includes('Exporter PNG')||!html.includes('JSON + provenance')||!html.includes('CSV des pixels'))throw new Error('Missing ergonomic/export controls');
 const points=[];
 for(const eventId of ['2020/fire_23756984','2020/fire_24695963']){
   const dates=events.find(e=>e.event_id===eventId).forecast_dates,mid=Math.floor(dates.length/2);
   for(const day of dates.slice(mid-1,mid+2)){
     const next=await requestForecast(base+'/lab',eventId,day);
     if(next.truthError||!next.truth||next.result.checkpoint_sha256!==status.checkpoint_sha256||next.result.target_pixels_read!==false)throw new Error('Additional event contract failed');
     if(next.result.input_dates.at(-1)!==day)throw new Error('Temporal context differs');
     const height=Math.ceil(next.result.full_shape[0]/next.result.preview_stride),width=Math.ceil(next.result.full_shape[1]/next.result.preview_stride);
     if(next.result.probability.length!==height||next.truth.length!==height||next.result.probability.some(r=>r.length!==width))throw new Error('Map dimensions differ');
     next.result.probability.forEach((row,y)=>row.forEach((p,x)=>{
       const prediction=next.result.predicted_fire[y][x];
       if(p===null?prediction!==null:prediction!==Number(p>=Math.fround(.78)))throw new Error('Threshold/mask differs');
     }));
     points.push({event_id:eventId,as_of:day,target_date:next.result.target_date,request_id:next.result.request_id,seconds:next.result.elapsed_seconds,tiles:next.result.tiles});
   }
 }
 const comparison=await (await fetch(base+'/science/propagation-comparison')).json();
 const comparisonHtml=renderToStaticMarkup(React.createElement(ComparisonView,{data:comparison}));
 if(!comparisonHtml.includes('FALSE')||comparison.promoted!==false)throw new Error('Experiment was promoted');
 const big=await (await fetch(base+'/science/big-data')).json();
 if(big.status!=='SPARK_EXECUTED')throw new Error('Big Data unavailable');
 fs.writeFileSync('../artifacts/prediction-lab-smoke.json',JSON.stringify({result,truth},null,2));
 fs.writeFileSync('../artifacts/prediction-lab-predicted.rgba',image.pixels);
 fs.writeFileSync('../artifacts/prediction-lab-errors.rgba',errors.pixels);
 fs.writeFileSync('../artifacts/prediction-lab-check.json',JSON.stringify({status:'PASS',path:'UI requestForecast -> FastAPI -> PyTorch CPU -> ForecastView -> imagePixels',event_id:result.event_id,as_of:result.as_of,target_date:result.target_date,request_id:result.request_id,model:result.model,checkpoint_sha256:result.checkpoint_sha256,threshold:result.threshold,target_pixels_read:result.target_pixels_read,elapsed_seconds:result.elapsed_seconds,tiles:result.tiles,canvas_maps:8,width:image.width,height:image.height,experiment_promoted:comparison.promoted,big_data:big.status,additional_points:points,exports:{png_renderer:true,csv_rows:image.width*image.height,json_provenance:true,invalid_cells_preserved:true},busy_retry_verified:true,browser_interactions_verified:'manual user confirmation 2026-11-13: eight maps, enlargement/close, date controls and PNG/JSON/CSV exports'},null,2));
 console.log(JSON.stringify({status:'PASS',maps:8,tiles:result.tiles,seconds:result.elapsed_seconds,request_id:result.request_id}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
`,resolveDir:process.cwd(),sourcefile:'lab-smoke-entry.mjs'},bundle:true,platform:'node',format:'cjs',minify:true,outfile:'../artifacts/prediction-lab-check.cjs',define:{'process.env.NODE_ENV':'"production"'}});
execFileSync(process.execPath,['../artifacts/prediction-lab-check.cjs'],{stdio:'inherit'});
