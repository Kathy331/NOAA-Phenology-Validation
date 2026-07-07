var SITE_NAMES = [
    'goodwater_AG_1000'
  ];
  
  var YEAR_FILTER = '2023'; // '2023', '2024', or 'ALL'
  
  // ---------
  // SPATIAL UNIFORMITY FUNCTION
  // ---------
  function uniformity(image, geom, scale, bandName) {
    var stats = image.reduceRegion({
      reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true}),
      geometry: geom,
      scale: scale,
      maxPixels: 1e9
    });
    var mean = ee.Number(stats.get(bandName + '_mean'));
    var std  = ee.Number(stats.get(bandName + '_stdDev'));
    return std.divide(mean);
  }
  
  // ---------
  // EMBEDDED SITE DATA
  // ---------
  var SITE_DATA = {
    "top_40_lowest_divergence_score": {
      "count": 40,
      "sites": [
        {
          "name": "utwente1_GR_1000", //0.41 vs 0.29 
          "lat": 52.240077,
          "lon": 6.850103,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "eslm1_EB_1002", //0.28 vs 0.14 
          "lat": 39.9426889,
          "lon": -5.7786833,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "eslm1_EB_1009", //0.28 vs 0.14 
          "lat": 39.9426889,
          "lon": -5.7786833,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "sevpjrm12_EN_1005", //0.26 vs 0.20 
          "lat": 34.38615,
          "lon": -106.52621,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "utwente2_AG_2000", //
          "lat": 52.237472,
          "lon": 6.86078,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "nistturf_AG_1000", //0.48 vs 0.42
          "lat": 39.1389,
          "lon": -77.2169,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "piedmontalpasture2_GR_1000", //0.24 vs 0.23 
          "lat": 33.8875,
          "lon": -85.6935,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "breedface2024control2_AG_1000", //0.38 vs 0.38 
          "lat": 50.625,
          "lon": 6.9844,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "tsubiology2_GR_1000", // 0.37/0.23 
          "lat": 36.1171,
          "lon": -86.8295,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "apvcontrol_AG_2000", //0.41 vs 0.29 
          "lat": 50.8632,
          "lon": 6.5311,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "alercecosteroforest_EN_1001", // 0.08 vs 0.12 
          "lat": -40.1726,
          "lon": -73.4439,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "NEON.D14.SRER.DP1.00042_SH_1000", //0.17 vs 0.09 desert
          "lat": 31.91068,
          "lon": -110.83549,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "eslm1_EB_1000", //0.288 vs 0.14 
          "lat": 39.9426889,
          "lon": -5.7786833,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "mead3_AG_1000", //0.3 vs 0.03
          "lat": 41.1797,
          "lon": -96.4397,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "sevmvecreo5ambinc_SH_1000", //0.07 vs 0.13
          "lat": 34.3384,
          "lon": -106.739,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "eslm1_EB_1000", //0.28 vs 0.14
          "lat": 39.9426889,
          "lon": -5.7786833,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "breedface2024ring1_AG_1000", //0.38 vs 0.33
          "lat": 50.6249,
          "lon": 6.986,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "bouldinalfalfa_AG_1000", //0.64 vs 0.13
          "lat": 38.0985,
          "lon": -121.4993,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "it25matschb1p_EN_1000",
          "lat": 46.6767,
          "lon": 10.5773,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "apvcontrol_AG_1000",
          "lat": 50.8632,
          "lon": 6.5311,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "congoflux_EB_1000",
          "lat": 0.8144,
          "lon": 24.5024,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "eucflux_EB_1000",
          "lat": -22.967875,
          "lon": -48.728009,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "macordway_GR_1000",
          "lat": 44.8106,
          "lon": -93.0272,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "NEON.D14.SRER.DP1.00033_SH_1000",
          "lat": 31.91068,
          "lon": -110.83549,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "breedface2024control1_AG_1000",
          "lat": 50.6249,
          "lon": 6.9871,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "NEON.D17.SJER.DP1.00042_GR_1000",
          "lat": 37.10878,
          "lon": -119.73228,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "cperuvb_GR_1000",
          "lat": 40.8055764,
          "lon": -104.7558542,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "rosemountcons_AG_2000",
          "lat": 44.6946,
          "lon": -93.0578,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "NEON.D10.ARIK.DP1.20002_GR_1000",
          "lat": 39.758246,
          "lon": -102.447103,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "aafcottawacfiaf14n_AG_1000",
          "lat": 45.2929,
          "lon": -75.767,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "mead3_AG_1000",
          "lat": 41.1797,
          "lon": -96.4397,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "saskatoon2_AG_1000",
          "lat": 52.1552,
          "lon": -106.6064,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "arsoljubljana_AG_1000",
          "lat": 46.0656,
          "lon": 14.5125,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "mead2_AG_2000",
          "lat": 41.1649,
          "lon": -96.4701,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "mead1_AG_1000",
          "lat": 41.1651,
          "lon": -96.4766,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "arkansascornsoy_AG_1000",
          "lat": 34.416,
          "lon": -91.6723,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "goodwaterbau_AG_1000",
          "lat": 39.23115,
          "lon": -92.15216,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "arscolessouth_AG_1000",
          "lat": 42.4816,
          "lon": -93.5235,
          "metadata": {
            "year": 2024
          }
        },
        {
          "name": "goodwater_AG_1000",
          "lat": 39.22848,
          "lon": -92.11936,
          "metadata": {
            "year": 2023
          }
        },
        {
          "name": "sevmvecreo22ambinc_SH_1000",
          "lat": 34.3386,
          "lon": -106.739,
          "metadata": {
            "year": 2023
          }
        }
      ]
    }
  };
  
  // ---------
  // LOOKUP LOGIC
  // ---------
  var allSites = SITE_DATA.top_40_lowest_divergence_score
    ? SITE_DATA.top_40_lowest_divergence_score.sites
    : [];
  
  var matchedSites = allSites.filter(function(site) {
    var nameMatches = SITE_NAMES.indexOf(site.name) !== -1;
    var yearMatches = (YEAR_FILTER === 'ALL') || (String(site.metadata.year) === String(YEAR_FILTER));
    return nameMatches && yearMatches;
  });
  
  // ---------
  // PER SITE LAYER GENERATION & UNIFORMITY MATH
  // ---------
  var allFootprintPoints = [];
  
  matchedSites.forEach(function(site, idx) {
    var label = site.name + ' (' + site.metadata.year + ')';
    var lat   = site.lat;
    var lon   = site.lon;
    var year  = String(site.metadata.year);
  
    var point           = ee.Geometry.Point([lon, lat]);
    var camera_fov      = point.buffer(200);
    var satellite_pixel = point.buffer(2000).bounds();
    var aoi             = satellite_pixel;
  
    var start = year + '-01-01';
    var end   = year + '-12-31';
  
    allFootprintPoints.push(point);
  
    var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(point)
      .filterDate(start, end)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
      .median();
  
    // 1: True Color
    Map.addLayer(
      s2.select(['B4','B3','B2']).clip(aoi),
      {min: 0, max: 2000},
      '1 ' + label + ' — True Color', false
    );
  
    // 2: NDVI
    var ndvi = s2.normalizedDifference(['B8','B4']).rename('NDVI');
    Map.addLayer(
      ndvi.clip(aoi),
      {min: -0.2, max: 0.9, palette: ['blue','white','yellow','green','darkgreen']},
      '2 ' + label + ' — NDVI', false
    );
  
    // 3: GVF
    var NDVI_bare = 0.05, NDVI_veg = 0.90;
    var gvf = ndvi.subtract(NDVI_bare).divide(NDVI_veg - NDVI_bare).pow(2).clamp(0,1).rename('GVF');
    Map.addLayer(
      gvf.clip(aoi),
      {min: 0, max: 1, palette: ['red','orange','yellow','lightgreen','darkgreen']},
      '3 ' + label + ' — GVF', false
    );
  
    // 4: Spatial CV
    var s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(point)
      .filterDate(start, end)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
      .map(function(img) {
        return img.normalizedDifference(['B8','B4']).rename('NDVI');
      });
  
    var ndvi_mean  = s2_collection.mean();
    var ndvi_std   = s2_collection.reduce(ee.Reducer.stdDev());
    var spatial_cv = ndvi_std.divide(ndvi_mean).rename('CV');
    Map.addLayer(
      spatial_cv.clip(aoi),
      {min: 0, max: 0.5, palette: ['darkgreen','yellow','orange','red']},
      '4 ' + label + ' — Spatial CV', false
    );
  
    // 5: Surface type flags
    var mndwi      = s2.normalizedDifference(['B3','B11']).rename('MNDWI');
    var water_mask = mndwi.gt(0.0);
  
    var bsi = s2.expression(
      '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
      {SWIR: s2.select('B11'), RED: s2.select('B4'), NIR: s2.select('B8'), BLUE: s2.select('B2')}
    ).rename('BSI');
    var bare_mask  = bsi.gt(0.0);
  
    var ndbi       = s2.normalizedDifference(['B11','B8']).rename('NDBI');
    var urban_mask = ndbi.gt(0.0);
  
    Map.addLayer(water_mask.clip(aoi).selfMask(), {palette: ['0000FF']}, '5a ' + label + ' — WATER', false);
    Map.addLayer(bare_mask.clip(aoi).selfMask(),  {palette: ['964B00']}, '5b ' + label + ' — BARE SOIL', false);
    Map.addLayer(urban_mask.clip(aoi).selfMask(), {palette: ['808080']}, '5c ' + label + ' — URBAN', false);
  
    // 6: Cloud frequency
    var s2_all = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(point)
      .filterDate(start, end);
  
    var cloud_freq = s2_all.map(function(img) {
      return img.select('SCL').eq(9).rename('cloud');
    }).mean();
    Map.addLayer(
      cloud_freq.clip(aoi),
      {min: 0, max: 0.5, palette: ['white','lightblue','blue','darkblue']},
      '⑥ ' + label + ' — Cloud frequency', false
    );
  
    // 7: UNIFORMITY PRINTOUT
    var cv_satellite = uniformity(ndvi, satellite_pixel, 10, 'NDVI');
    var cv_camera    = uniformity(ndvi, camera_fov, 10, 'NDVI');
  
    print(label + ' - satellite 4km uniformity (CV):', cv_satellite);
    print(label + ' - camera 200m uniformity (CV):', cv_camera);
  
    // 8: Footprints
    Map.addLayer(
      ee.Image().byte().paint({featureCollection: ee.FeatureCollection([ee.Feature(satellite_pixel)]), color: 1, width: 2}),
      {palette: ['FFA500']}, label + ' — VIIRS pixel', true
    );
    Map.addLayer(
      ee.Image().byte().paint({featureCollection: ee.FeatureCollection([ee.Feature(camera_fov)]), color: 1, width: 2}),
      {palette: ['FF0000']}, label + ' — PhenoCam FOV', true
    );
    Map.addLayer(point, {color: 'white'}, label + ' — Camera', true);
  });
  
  // ---------
  // MAP VIEW
  // ---------
  if (allFootprintPoints.length > 0) {
    var allPointsFC = ee.FeatureCollection(allFootprintPoints.map(function(p) {
      return ee.Feature(p);
    }));
    Map.centerObject(allPointsFC, allFootprintPoints.length === 1 ? 13 : 4);
  }
  
  Map.setOptions('SATELLITE');