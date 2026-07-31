#!/usr/bin/env python3
from __future__ import annotations
import base64, gzip, io, json, math, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import transform as shp_transform
from shapely.validation import make_valid
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output'; OUT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'cache'; CACHE.mkdir(parents=True,exist_ok=True)
IDS_B64='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'
IDS=json.loads(gzip.decompress(base64.b64decode(IDS_B64)))
GHSL='https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_1000/V1-0/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.zip'
ARCGIS='https://services7.arcgis.com/poOcx60xJtGtoR7g/ArcGIS/rest/services/Lakes/FeatureServer/0/query'
RADS=(10,25,50); SIGMAS=(5,10,20,30); MAXKM=90
S=requests.Session(); S.headers['User-Agent']='academic-lake-population/1.0'

def download(url,dest):
    if dest.exists() and dest.stat().st_size>10_000_000:return
    for k in range(6):
        try:
            with S.get(url,stream=True,timeout=(30,300)) as r:
                r.raise_for_status()
                with open(str(dest)+'.part','wb') as f:
                    for b in r.iter_content(1024*1024):
                        if b:f.write(b)
            Path(str(dest)+'.part').replace(dest);return
        except Exception as e:
            print('download retry',k,e,flush=True);time.sleep(2**k)
    raise RuntimeError('download failed')

def geometries():
    p=CACHE/'lakes.geojson'
    if p.exists():return gpd.read_file(p)
    feats=[]
    for i in range(0,len(IDS),25):
        ids=IDS[i:i+25]
        data={'where':'Hylak_id IN ('+','.join(map(str,ids))+')','outFields':'Hylak_id,Lake_area,Shore_len','returnGeometry':'true','outSR':'4326','f':'geojson','geometryPrecision':'5','maxAllowableOffset':'0.0001'}
        for k in range(6):
            try:
                j=S.post(ARCGIS,data=data,timeout=(30,300)).json()
                if 'error' in j:raise RuntimeError(j['error'])
                feats+=j.get('features',[]);break
            except Exception as e:
                if k==5:raise
                time.sleep(2**k)
        if i%250==0:print('geometry',i,len(feats),flush=True)
    p.write_text(json.dumps({'type':'FeatureCollection','features':feats}))
    return gpd.GeoDataFrame.from_features(feats,crs='EPSG:4326')

def validgeom(g):
    if g is None or g.is_empty:return None
    try:g=make_valid(g)
    except:g=g.buffer(0)
    return None if g.is_empty else g

def window(win,w,h):
    c0=max(0,math.floor(win.col_off));r0=max(0,math.floor(win.row_off));c1=min(w,math.ceil(win.col_off+win.width));r1=min(h,math.ceil(win.row_off+win.height))
    return None if c1<=c0 or r1<=r0 else Window(c0,r0,c1-c0,r1-r0)

def main():
    z=CACHE/'pop.zip';download(GHSL,z)
    d=CACHE/'pop';d.mkdir(exist_ok=True)
    if not list(d.rglob('*.tif')):
        with zipfile.ZipFile(z) as q:q.extractall(d)
    tif=list(d.rglob('*.tif'))[0]
    g=geometries();g['Hylak_id']=pd.to_numeric(g.Hylak_id).astype(int);g=g.drop_duplicates('Hylak_id').set_index('Hylak_id')
    trf=Transformer.from_crs('EPSG:4326','ESRI:54009',always_xy=True)
    rows=[]
    with rasterio.open(tif) as src:
        px=abs(src.transform.a*src.transform.e)/1e6; sy=abs(src.transform.e)/1000;sx=abs(src.transform.a)/1000;nd=src.nodata
        print('raster',src.width,src.height,src.crs,px,flush=True)
        for n,hid in enumerate(IDS,1):
            r={'hylak_id':hid,'spatial_status':'OK'}
            try:
                if hid not in g.index:raise KeyError('geometry')
                gm=validgeom(shp_transform(trf.transform,validgeom(g.loc[hid].geometry)))
                if gm is None:raise ValueError('geometry')
                w=window(from_bounds(*gm.buffer(MAXKM*1000).bounds,transform=src.transform),src.width,src.height)
                if w is None:raise ValueError('window')
                a=src.read(1,window=w,masked=False);t=src.window_transform(w)
                land=np.isfinite(a)&(a>=0)
                if nd is not None:land&=a!=nd
                pop=np.where(land,a,0).astype(float)
                lm=rasterize([(mapping(gm),1)],out_shape=a.shape,transform=t,fill=0,all_touched=True,dtype='uint8').astype(bool)
                if not lm.any():lm=rasterize([(mapping(gm.buffer(500)),1)],out_shape=a.shape,transform=t,fill=0,all_touched=True,dtype='uint8').astype(bool)
                dist=distance_transform_edt(~lm,sampling=(sy,sx));land&=(~lm)&(dist>0)
                for rad in RADS:
                    m=land&(dist<=rad);ar=float(m.sum()*px);pp=float(pop[m].sum())
                    r[f'population_mean_{rad}km']=pp;r[f'land_area_mean_{rad}km_km2']=ar;r[f'density_mean_{rad}km']=pp/ar if ar else np.nan
                for sig in SIGMAS:
                    m=land&(dist<=3*sig);ww=np.exp(-(dist[m]**2)/(2*sig*sig));ar=float(ww.sum()*px);pp=float((pop[m]*ww).sum())
                    r[f'population_gaussian_sigma{sig}km']=pp;r[f'effective_land_area_gaussian_sigma{sig}km_km2']=ar;r[f'density_gaussian_sigma{sig}km']=pp/ar if ar else np.nan
            except Exception as e:r['spatial_status']='ERROR:'+type(e).__name__
            rows.append(r)
            if n%100==0:
                print('processed',n,len(IDS),flush=True);pd.DataFrame(rows).to_csv(OUT/'partial.csv',index=False)
    df=pd.DataFrame(rows);df.to_csv(OUT/'lake_population_metrics.csv',index=False)
    summary={'n_target':len(IDS),'n_rows':len(df),'n_ok':int((df.spatial_status=='OK').sum()),'n_error':int((df.spatial_status!='OK').sum()),'population_source':'GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0','geometry_source':'HydroLAKES ArcGIS FeatureServer','mean_buffers_km':RADS,'gaussian_sigma_km':SIGMAS,'land_denominator':'valid non-water GHSL 1 km cells outside HydroLAKES polygon'}
    (OUT/'spatial_summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':main()
