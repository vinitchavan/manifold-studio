'use strict';
const $ = id => document.getElementById(id);
let catalog, lastRequest = null, busy = false;
const names = {sphere:'Sphere · S²', torus:'Torus · S¹ × S¹', cylinder:'Cylinder', mobius:'Möbius strip', plane:'Plane · R²', circle:'Circle · S¹'};
function factor(value='sphere') {
  const row=document.createElement('div'); row.className='factor';
  const select=document.createElement('select'); select.setAttribute('aria-label','Geometry factor');
  for(const [key,label] of Object.entries(names)){const opt=new Option(label,key);select.add(opt);}select.value=value;
  const remove=document.createElement('button');remove.type='button';remove.textContent='×';remove.setAttribute('aria-label','Remove geometry factor');
  remove.onclick=()=>{if($('factors').children.length>1){row.remove();sync();}};select.onchange=sync;
  row.append(select,remove);$('factors').append(row);sync();
}
function sync(){
  $('vector-fields').hidden=$('source').value!=='vectors';$('text-fields').hidden=$('source').value!=='text';
  $('training-fields').hidden=$('mode').value!=='train';$('add-factor').disabled=$('factors').children.length>=3;
  const mobius=[...$('factors').querySelectorAll('select')].some(s=>s.value==='mobius');
  const ambientLoss=$('mode').value==='train'&&['euclidean_triplet','ambient_triplet'].includes($('loss').value);
  if(mobius||ambientLoss)$('metric').value='ambient';
  $('geometry-note').textContent=mobius?'Möbius uses ambient distance. Intrinsic Möbius geodesics are not implemented.':'Product coordinates retain every factor. The 3D view is a projection.';
  $('run').firstChild.textContent=$('mode').value==='train'?'Train & explore ':'Explore geometry ';
}
function request(){
  const source=$('source').value;
  return {source,mode:$('mode').value,geometry:[...$('factors').querySelectorAll('select')].map(s=>s.value),metric:$('metric').value,
    vectors:source==='vectors'?JSON.parse($('vectors').value):null,texts:source==='text'?$('texts').value.split('\n').map(s=>s.trim()).filter(Boolean):null,
    loss:$('loss').value,steps:Number($('steps').value),margin:Number($('margin').value),alpha:Number($('alpha').value),beta:Number($('beta').value),gamma:Number($('gamma').value),
    seed:Number($('seed').value),target:$('arrows').checked?Number($('target').value):null,show_labels:$('show-labels').checked,
    query_index:Number($('query').value),triplets:$('triplets').value.trim()?JSON.parse($('triplets').value):null};
}
function error(message){$('error').textContent=message;$('error').hidden=false;}
function setBusy(value,message){busy=value;$('run').disabled=value;$('export').disabled=value||!lastRequest;$('run-status').textContent=message;}
async function call(path,data){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!response.ok){let message=`Request failed (${response.status})`;try{const body=await response.json();message=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail);}catch{}throw new Error(message);}return response;}
$('controls').onsubmit=async event=>{event.preventDefault();if(busy)return;$('error').hidden=true;
 try{const data=request();setBusy(true,data.mode==='train'?'Training… this can take a moment.':'Mapping and generating the plot…');
 const response=await (await call('/api/run',data)).json();
 $('empty').hidden=true;$('plot').hidden=false;
 await Plotly.react('plot',response.plot.data,response.plot.layout,{responsive:true,displaylogo:false});
 const r=response.report;$('rows').textContent=r.rows;$('dimensions').textContent=`${r.intrinsic_dim} / ${r.ambient_dim}`;$('overlap').textContent=`${(r.metrics.neighbor_overlap*100).toFixed(1)}%`;
 $('view-title').textContent=data.geometry.map(g=>g[0].toUpperCase()+g.slice(1)).join(' × ');
 $('details').textContent=JSON.stringify(r,null,2);$('loss-panel').hidden=!r.history.length;
 if(r.history.length){const keys=Object.keys(r.history[0]).filter(k=>k!=='step');await Plotly.react('loss-plot',keys.map(k=>({x:r.history.map(h=>h.step),y:r.history.map(h=>h[k]),name:k.replaceAll('_',' '),mode:'lines',line:{width:k==='total'?3:1.5}})),{paper_bgcolor:'#101a2b',plot_bgcolor:'#101a2b',font:{color:'#96a7be'},margin:{t:15,b:45,l:45,r:10},xaxis:{title:{text:'Step'},gridcolor:'#233148'},yaxis:{title:{text:'Objective'},gridcolor:'#233148'},legend:{orientation:'h'},height:260},{responsive:true,displaylogo:false});}
 lastRequest=data;setBusy(false,'Run complete. Download replays this exact configuration.');
 }catch(e){error(e.message);setBusy(false,'Run failed. Your previous successful result is unchanged.');}};
$('export').onclick=async()=>{if(!lastRequest||busy)return;$('error').hidden=true;setBusy(true,'Replaying the run and preparing your ZIP…');try{const blob=await (await call('/api/export',lastRequest)).blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='manifold-studio-run.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),10000);setBusy(false,'Downloaded embeddings, model, metrics and an offline 3D explorer.');}catch(e){error(e.message);setBusy(false,'Download failed. Try again.');}};
$('vector-file').onchange=async()=>{const file=$('vector-file').files[0];if(!file)return;try{if(file.size>8*1024*1024)throw new Error('Choose a file smaller than 8 MB.');const text=await file.text();let values;if(file.name.toLowerCase().endsWith('.csv')){values=text.trim().split(/\r?\n/).map(line=>line.split(',').map(v=>{if(!v.trim()||!Number.isFinite(Number(v)))throw new Error('CSV must contain numeric values without headers.');return Number(v);}));}else{values=JSON.parse(text);}if(!Array.isArray(values))throw new Error('The file must contain an array of numeric rows.');$('vectors').value=JSON.stringify(values);$('error').hidden=true;}catch(e){error(e.message);}};
for(const id of ['source','mode','loss'])$(id).onchange=sync;
$('add-factor').onclick=()=>{if($('factors').children.length<3)factor('torus');};
(async()=>{try{const r=await fetch('/api/catalog');if(!r.ok)throw new Error('Backend is unavailable.');catalog=await r.json();for(const name of catalog.losses)$('loss').add(new Option(name.replaceAll('_',' '),name));$('loss').value='geodesic_triplet';$('mode').options[1].disabled=!catalog.training;$('source').options[2].disabled=!catalog.text;factor('sphere');$('connection').textContent='● Backend connected';$('connection').title=`Training: ${catalog.training?'ready':'install [train]'} · Sentences: ${catalog.text?'ready':'install [text]'}`;}catch(e){$('connection').textContent='Backend unavailable';$('run').disabled=true;error(e.message);}})();
