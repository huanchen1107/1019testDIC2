/**
 * Interactive Feature Selection Studio — 50 Startups & Spotify Tracks
 * Inspired by interactive-ml.com design & UX
 */

// =========================================================================
// 1. DATASETS DEFINITIONS
// =========================================================================

const PROBLEMS_DATA = {
  startup50: {
    id: "startup50",
    name: "50 Startups Profit Prediction",
    badge: "50 Startups Studio",
    title: "Feature Selection & Regression Studio: 50 Startups",
    subtitle: "Explore how feature subset size ($k$) impacts model complexity, validation error, and predictive power on the Kaggle 50 Startups dataset.",
    targetName: "Profit",
    targetUnit: "$",
    maxK: 5,
    defaultK: 2,
    algorithms: {
      sfs: {
        name: "Sequential Forward Selection (SFS)",
        optimalK: 2,
        records: [
          { k: 1, features: ["R&D Spend"], rmse: 8274.868018, mae: 6772.453280, r2: 0.946459, modelKey: "R&D Spend" },
          { k: 2, features: ["R&D Spend", "Marketing Spend"], rmse: 8198.797191, mae: 6886.594588, r2: 0.947439, modelKey: "Marketing Spend__R&D Spend" },
          { k: 3, features: ["R&D Spend", "Marketing Spend", "State_New York"], rmse: 8309.059683, mae: 7028.435332, r2: 0.946015, modelKey: "Marketing Spend__R&D Spend__State_New York" },
          { k: 4, features: ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida"], rmse: 8409.916714, mae: 7109.161998, r2: 0.944697, modelKey: "Marketing Spend__R&D Spend__State_Florida__State_New York" },
          { k: 5, features: ["R&D Spend", "Marketing Spend", "State_New York", "State_Florida", "Administration"], rmse: 9137.990153, mae: 7514.293660, r2: 0.934707, modelKey: "Administration__Marketing Spend__R&D Spend__State_Florida__State_New York" }
        ]
      },
      rfe: {
        name: "Recursive Feature Elimination (RFE)",
        optimalK: 3,
        records: [
          { k: 1, features: ["R&D Spend"], rmse: 8274.868018, mae: 6772.453280, r2: 0.946459, modelKey: "R&D Spend" },
          { k: 2, features: ["R&D Spend", "State_Florida"], rmse: 8217.446448, mae: 6772.403035, r2: 0.947199, modelKey: "R&D Spend__State_Florida" },
          { k: 3, features: ["R&D Spend", "Administration", "State_Florida"], rmse: 8171.645989, mae: 6730.785004, r2: 0.947786, modelKey: "Administration__R&D Spend__State_Florida" },
          { k: 4, features: ["R&D Spend", "Administration", "State_Florida", "State_New York"], rmse: 8178.369215, mae: 6751.890337, r2: 0.947700, modelKey: "Administration__R&D Spend__State_Florida__State_New York" },
          { k: 5, features: ["R&D Spend", "Administration", "Marketing Spend", "State_Florida", "State_New York"], rmse: 9137.990153, mae: 7514.293660, r2: 0.934707, modelKey: "Administration__Marketing Spend__R&D Spend__State_Florida__State_New York" }
        ]
      },
      exhaustive: {
        name: "Exhaustive Best Subset Selection",
        optimalK: 3,
        records: [
          { k: 1, features: ["R&D Spend"], rmse: 8274.868018, mae: 6772.453280, r2: 0.946459, modelKey: "R&D Spend" },
          { k: 2, features: ["R&D Spend", "Marketing Spend"], rmse: 8198.797191, mae: 6886.594588, r2: 0.947439, modelKey: "Marketing Spend__R&D Spend" },
          { k: 3, features: ["R&D Spend", "Administration", "State_Florida"], rmse: 8171.645989, mae: 6730.785004, r2: 0.947786, modelKey: "Administration__R&D Spend__State_Florida" },
          { k: 4, features: ["R&D Spend", "Administration", "State_Florida", "State_New York"], rmse: 8178.369215, mae: 6751.890337, r2: 0.947700, modelKey: "Administration__R&D Spend__State_Florida__State_New York" },
          { k: 5, features: ["R&D Spend", "Administration", "Marketing Spend", "State_Florida", "State_New York"], rmse: 9137.990153, mae: 7514.293660, r2: 0.934707, modelKey: "Administration__Marketing Spend__R&D Spend__State_Florida__State_New York" }
        ]
      }
    },
    models: {"R&D Spend": {"intercept": 48416.297661385026, "coef": {"R&D Spend": 0.8516227998779371}}, "Marketing Spend__R&D Spend": {"intercept": 46346.338761236555, "coef": {"R&D Spend": 0.7886182999424615, "Marketing Spend": 0.032576627317020684}}, "Marketing Spend__R&D Spend__State_New York": {"intercept": 46100.29130551415, "coef": {"R&D Spend": 0.7858709815066744, "Marketing Spend": 0.033271829410322355, "State_New York": 864.0018700527347}}, "Marketing Spend__R&D Spend__State_Florida__State_New York": {"intercept": 46329.060178542495, "coef": {"R&D Spend": 0.7851417173112172, "Marketing Spend": 0.033956402340301096, "State_New York": 554.657332826717, "State_Florida": -826.4681587619036}}, "Administration__Marketing Spend__R&D Spend__State_Florida__State_New York": {"intercept": 42554.16761773237, "coef": {"R&D Spend": 0.7734671927328001, "Administration": 0.03288459753625596, "Marketing Spend": 0.036610025863856116, "State_Florida": -959.2841600571066, "State_New York": 699.3690525192583}}, "R&D Spend__State_Florida": {"intercept": 48319.70480319308, "coef": {"R&D Spend": 0.8515781310728888, "State_Florida": 399.1758990382208}}, "Administration__R&D Spend__State_Florida": {"intercept": 48524.91882908291, "coef": {"R&D Spend": 0.851939574961828, "Administration": -0.0018828428042859105, "State_Florida": 404.0171892528541}}, "Administration__R&D Spend__State_Florida__State_New York": {"intercept": 48444.527070978846, "coef": {"R&D Spend": 0.8516135792403573, "Administration": -0.001696797748794788, "State_Florida": 484.73296619775977, "State_New York": 173.19542846407654}}},
    testSamples: [{"rd": 66051.52, "admin": 182645.56, "marketing": 118148.2, "state_ny": 0.0, "state_fl": 1.0, "profit": 103282.38}, {"rd": 100671.96, "admin": 91790.61, "marketing": 249744.55, "state_ny": 0.0, "state_fl": 0.0, "profit": 144259.4}, {"rd": 101913.08, "admin": 110594.11, "marketing": 229160.95, "state_ny": 0.0, "state_fl": 1.0, "profit": 146121.95}, {"rd": 27892.92, "admin": 84710.77, "marketing": 164470.71, "state_ny": 0.0, "state_fl": 1.0, "profit": 77798.83}, {"rd": 153441.51, "admin": 101145.55, "marketing": 407934.54, "state_ny": 0.0, "state_fl": 1.0, "profit": 191050.39}, {"rd": 72107.6, "admin": 127864.55, "marketing": 353183.81, "state_ny": 1.0, "state_fl": 0.0, "profit": 105008.31}, {"rd": 20229.59, "admin": 65947.93, "marketing": 185265.1, "state_ny": 1.0, "state_fl": 0.0, "profit": 81229.06}, {"rd": 61136.38, "admin": 152701.92, "marketing": 88218.23, "state_ny": 1.0, "state_fl": 0.0, "profit": 97483.56}, {"rd": 73994.56, "admin": 122782.75, "marketing": 303319.26, "state_ny": 0.0, "state_fl": 1.0, "profit": 110352.25}, {"rd": 142107.34, "admin": 91391.77, "marketing": 366168.42, "state_ny": 0.0, "state_fl": 1.0, "profit": 166187.94}]
  },
  spotify: {
    id: "spotify",
    name: "Spotify Tracks Popularity Prediction",
    badge: "Spotify Popularity Studio",
    title: "Feature Selection & Popularity Studio: Spotify Tracks",
    subtitle: "Investigate how acoustic dimensions (danceability, energy, loudness, speechiness, valence, tempo) linearly impact track Popularity (0-100 score).",
    targetName: "Popularity",
    targetUnit: "pts",
    maxK: 6,
    defaultK: 6,
    algorithms: {
      sfs: {
        name: "Sequential Forward Selection (SFS)",
        optimalK: 6,
        records: [{"k": 1, "features": ["danceability"], "rmse": 22.413983753541867, "mae": 19.120827611248895, "r2": 0.00512825693781882, "modelKey": "danceability"}, {"k": 2, "features": ["danceability", "speechiness"], "rmse": 22.362006640701658, "mae": 19.013579312809295, "r2": 0.009737040818395548, "modelKey": "danceability__speechiness"}, {"k": 3, "features": ["danceability", "speechiness", "instrumentalness"], "rmse": 22.323048058877244, "mae": 18.810660124949113, "r2": 0.013184462617563453, "modelKey": "danceability__instrumentalness__speechiness"}, {"k": 4, "features": ["danceability", "speechiness", "instrumentalness", "valence"], "rmse": 22.278816615886697, "mae": 18.695009702208377, "r2": 0.017091190066210404, "modelKey": "danceability__instrumentalness__speechiness__valence"}, {"k": 5, "features": ["danceability", "speechiness", "instrumentalness", "valence", "loudness"], "rmse": 22.26549442971872, "mae": 18.68928659387397, "r2": 0.01826634934729221, "modelKey": "danceability__instrumentalness__loudness__speechiness__valence"}, {"k": 6, "features": ["danceability", "speechiness", "instrumentalness", "valence", "loudness", "energy"], "rmse": 22.258465721932968, "mae": 18.684741633143954, "r2": 0.018886073273850612, "modelKey": "danceability__energy__instrumentalness__loudness__speechiness__valence"}]
      },
      rfe: {
        name: "Recursive Feature Elimination (RFE)",
        optimalK: 6,
        records: [{"k": 1, "features": ["speechiness"], "rmse": 22.43718023060188, "mae": 19.136517595255654, "r2": 0.0030679840824223703, "modelKey": "speechiness"}, {"k": 2, "features": ["danceability", "speechiness"], "rmse": 22.362006640701658, "mae": 19.013579312809295, "r2": 0.009737040818395548, "modelKey": "danceability__speechiness"}, {"k": 3, "features": ["danceability", "speechiness", "valence"], "rmse": 22.345026052309326, "mae": 18.972231056856923, "r2": 0.011240382057081555, "modelKey": "danceability__speechiness__valence"}, {"k": 4, "features": ["danceability", "instrumentalness", "speechiness", "valence"], "rmse": 22.278816615886697, "mae": 18.695009702208385, "r2": 0.017091190066210404, "modelKey": "danceability__instrumentalness__speechiness__valence"}, {"k": 5, "features": ["danceability", "energy", "instrumentalness", "speechiness", "valence"], "rmse": 22.27957613419543, "mae": 18.69896679749444, "r2": 0.017024171249626296, "modelKey": "danceability__energy__instrumentalness__speechiness__valence"}, {"k": 6, "features": ["danceability", "energy", "loudness", "instrumentalness", "speechiness", "valence"], "rmse": 22.258465721932968, "mae": 18.684741633143954, "r2": 0.01888607327385039, "modelKey": "danceability__energy__instrumentalness__loudness__speechiness__valence"}]
      },
      exhaustive: {
        name: "Exhaustive Best Subset Selection",
        optimalK: 6,
        records: [{"k": 1, "features": ["danceability"], "rmse": 22.413983753541867, "mae": 19.120827611248895, "r2": 0.00512825693781882, "modelKey": "danceability"}, {"k": 2, "features": ["danceability", "speechiness"], "rmse": 22.362006640701658, "mae": 19.013579312809295, "r2": 0.009737040818395548, "modelKey": "danceability__speechiness"}, {"k": 3, "features": ["danceability", "instrumentalness", "speechiness"], "rmse": 22.323048058877244, "mae": 18.810660124949113, "r2": 0.013184462617563453, "modelKey": "danceability__instrumentalness__speechiness"}, {"k": 4, "features": ["danceability", "instrumentalness", "speechiness", "valence"], "rmse": 22.278816615886697, "mae": 18.695009702208385, "r2": 0.017091190066210404, "modelKey": "danceability__instrumentalness__speechiness__valence"}, {"k": 5, "features": ["danceability", "loudness", "instrumentalness", "speechiness", "valence"], "rmse": 22.26549442971872, "mae": 18.689286593873966, "r2": 0.0182663493472921, "modelKey": "danceability__instrumentalness__loudness__speechiness__valence"}, {"k": 6, "features": ["danceability", "energy", "loudness", "instrumentalness", "speechiness", "valence"], "rmse": 22.258465721932968, "mae": 18.684741633143954, "r2": 0.01888607327385039, "modelKey": "danceability__energy__instrumentalness__loudness__speechiness__valence"}]
      }
    },
    models: {"danceability": {"intercept": 28.708640428648543, "coef": {"danceability": 7.581633987277098}}, "energy": {"intercept": 32.182254206661824, "coef": {"energy": 1.2207994361540837}}, "loudness": {"intercept": 35.952679473374936, "coef": {"loudness": 0.3682848014670881}}, "instrumentalness": {"intercept": 34.12172410454925, "coef": {"instrumentalness": -7.714562683597093}}, "speechiness": {"intercept": 33.90953423942861, "coef": {"speechiness": -10.980259541720978}}, "valence": {"intercept": 34.415303809687096, "coef": {"valence": -3.070537925407511}}, "danceability__energy": {"intercept": 28.344377027818894, "coef": {"danceability": 7.479154583790565, "energy": 0.652566377544453}}, "danceability__loudness": {"intercept": 32.52522945807897, "coef": {"danceability": 5.413925591634688, "loudness": 0.3209197491592741}}, "danceability__instrumentalness": {"intercept": 30.987228772218206, "coef": {"danceability": 5.42818697745579, "instrumentalness": -7.161665553169419}}, "danceability__speechiness": {"intercept": 29.31057008860399, "coef": {"danceability": 8.403096700010705, "speechiness": -12.452498629041}}, "danceability__valence": {"intercept": 29.192523060428318, "coef": {"danceability": 12.729981143540886, "valence": -7.18526761028633}}, "energy__loudness": {"intercept": 45.13724050730755, "coef": {"energy": -9.549321654209475, "loudness": 0.7402452201233418}}, "energy__instrumentalness": {"intercept": 34.24864972602236, "coef": {"energy": -0.19071078567523173, "instrumentalness": -7.7389175837190765}}, "energy__speechiness": {"intercept": 32.733767097671084, "coef": {"energy": 1.8993245916982398, "speechiness": -11.590643531459255}}, "energy__valence": {"intercept": 33.28739289918538, "coef": {"energy": 2.093773719764022, "valence": -3.550559718078837}}, "instrumentalness__loudness": {"intercept": 35.59981543650672, "coef": {"loudness": 0.20793656228932758, "instrumentalness": -6.338911092930071}}, "loudness__speechiness": {"intercept": 37.10686174093405, "coef": {"loudness": 0.3840684303036967, "speechiness": -12.014838935001745}}, "loudness__valence": {"intercept": 39.018842129928295, "coef": {"loudness": 0.4423488159238721, "valence": -5.245786370860812}}, "instrumentalness__speechiness": {"intercept": 35.30294080295629, "coef": {"instrumentalness": -8.144044326469047, "speechiness": -13.077292031308035}}, "instrumentalness__valence": {"intercept": 37.59293924551569, "coef": {"instrumentalness": -9.658377689993914, "valence": -6.765877067286728}}, "speechiness__valence": {"intercept": 35.23648199910775, "coef": {"speechiness": -10.674102400647303, "valence": -2.877643435792926}}, "danceability__energy__loudness": {"intercept": 41.853365356494784, "coef": {"danceability": 4.396161525439914, "energy": -9.028682948229719, "loudness": 0.6815046662565667}}, "danceability__energy__instrumentalness": {"intercept": 31.292633020134755, "coef": {"danceability": 5.4910214608343235, "energy": -0.5133996213748357, "instrumentalness": -7.220829622156806}}, "danceability__energy__speechiness": {"intercept": 28.576992138518122, "coef": {"danceability": 8.217717614319124, "energy": 1.3489083724527995, "speechiness": -12.853517143927258}}, "danceability__energy__valence": {"intercept": 28.030893677066505, "coef": {"danceability": 12.750122498813873, "energy": 2.141026569956285, "valence": -7.682632982835363}}, "danceability__instrumentalness__loudness": {"intercept": 32.71842979009451, "coef": {"danceability": 4.572609806305791, "loudness": 0.17404111697116625, "instrumentalness": -6.09740323956541}}, "danceability__loudness__speechiness": {"intercept": 33.27224829037339, "coef": {"danceability": 6.200239974237615, "loudness": 0.33106358667465474, "speechiness": -12.958352070954195}}, "danceability__loudness__valence": {"intercept": 33.95469748067103, "coef": {"danceability": 11.079382213003038, "loudness": 0.3925207975474545, "valence": -8.581961573876296}}, "danceability__instrumentalness__speechiness": {"intercept": 31.78483191219653, "coef": {"danceability": 6.239143300716387, "instrumentalness": -7.539331800578216, "speechiness": -14.01469312215425}}, "danceability__instrumentalness__valence": {"intercept": 32.46407197168611, "coef": {"danceability": 12.400601301750344, "instrumentalness": -9.533082421487151, "valence": -10.726202304850245}}, "danceability__speechiness__valence": {"intercept": 29.80481773865076, "coef": {"danceability": 13.598872951882413, "speechiness": -12.592687281324228, "valence": -7.238554992296956}}, "energy__instrumentalness__loudness": {"intercept": 42.706840769245, "coef": {"energy": -7.323110985888026, "loudness": 0.522101610560897, "instrumentalness": -5.195685579089095}}, "energy__loudness__speechiness": {"intercept": 45.22385472362478, "coef": {"energy": -8.626505842009976, "loudness": 0.7176222372273039, "speechiness": -10.141053905129718}}, "energy__loudness__valence": {"intercept": 47.55002714729116, "coef": {"energy": -9.093997705532429, "loudness": 0.791369565266442, "valence": -4.8771904990628165}}, "energy__instrumentalness__speechiness": {"intercept": 34.96985081433786, "coef": {"energy": 0.5210556671858193, "instrumentalness": -8.082481207051313, "speechiness": -13.228891097111864}}, "energy__instrumentalness__valence": {"intercept": 36.920435192106694, "coef": {"energy": 1.197573217302307, "instrumentalness": -9.575174147839927, "valence": -7.008600497486166}}, "energy__speechiness__valence": {"intercept": 33.816638424587325, "coef": {"energy": 2.7526666256788745, "speechiness": -11.493157095753304, "valence": -3.493922681320244}}, "instrumentalness__loudness__speechiness": {"intercept": 36.85528626584925, "coef": {"loudness": 0.2156459794246838, "instrumentalness": -6.724462189482012, "speechiness": -13.292653485011403}}, "instrumentalness__loudness__valence": {"intercept": 39.879803620591815, "coef": {"loudness": 0.26955506558722475, "instrumentalness": -8.082696591276655, "valence": -7.488550230687514}}, "loudness__speechiness__valence": {"intercept": 40.0543391028685, "coef": {"loudness": 0.4556178180158231, "speechiness": -11.664951791522672, "valence": -5.100236245454802}}, "instrumentalness__speechiness__valence": {"intercept": 38.71360367721668, "coef": {"instrumentalness": -10.057828664412169, "speechiness": -12.858707571390728, "valence": -6.686335763465664}}, "danceability__energy__instrumentalness__loudness": {"intercept": 39.833648839559494, "coef": {"danceability": 3.9370754519869413, "energy": -6.918901757301849, "loudness": 0.4755764252800905, "instrumentalness": -5.050846144147752}}, "danceability__energy__loudness__speechiness": {"intercept": 41.35277007605236, "coef": {"danceability": 5.193041625122704, "energy": -7.925730278837464, "loudness": 0.64613147242091, "speechiness": -11.083515216489657}}, "danceability__energy__loudness__valence": {"intercept": 41.750572092971574, "coef": {"danceability": 9.745787849975546, "energy": -7.660407778031361, "loudness": 0.6925191101746684, "valence": -7.869905534369138}}, "danceability__energy__instrumentalness__speechiness": {"intercept": 31.668460624610002, "coef": {"danceability": 6.217750471859824, "energy": 0.20091079063914252, "instrumentalness": -7.517667482956586, "speechiness": -14.069933145438656}}, "danceability__energy__instrumentalness__valence": {"intercept": 31.752789108447818, "coef": {"danceability": 12.415433648482741, "energy": 1.2557048994747868, "instrumentalness": -9.445690211151595, "valence": -10.985444770982696}}, "danceability__energy__speechiness__valence": {"intercept": 28.26503996266591, "coef": {"danceability": 13.687057978633247, "energy": 2.9168974142162405, "speechiness": -13.473050278393488, "valence": -7.919882293467923}}, "danceability__instrumentalness__loudness__speechiness": {"intercept": 33.5405889078993, "coef": {"danceability": 5.375406582713175, "loudness": 0.17624521562660367, "instrumentalness": -6.462837493548797, "speechiness": -14.060933535454568}}, "danceability__instrumentalness__loudness__valence": {"intercept": 34.624486392877046, "coef": {"danceability": 11.5474071063094, "loudness": 0.21305599073233872, "instrumentalness": -8.296286674383841, "valence": -11.024921420344349}}, "danceability__loudness__speechiness__valence": {"intercept": 34.73374632636167, "coef": {"danceability": 11.945823819062781, "loudness": 0.40368173826317744, "speechiness": -13.237370340144274, "valence": -8.677690616911711}}, "danceability__instrumentalness__speechiness__valence": {"intercept": 33.334010758380586, "coef": {"danceability": 13.401690340894076, "instrumentalness": -9.980633489695524, "speechiness": -14.732705983308607, "valence": -10.954782217156417}}, "energy__instrumentalness__loudness__speechiness": {"intercept": 42.55202349302777, "coef": {"energy": -6.015268582971151, "loudness": 0.47283789000101584, "instrumentalness": -5.742094656570024, "speechiness": -11.799387678319981}}, "energy__instrumentalness__loudness__valence": {"intercept": 45.2871560687861, "coef": {"energy": -5.882293191302089, "loudness": 0.5175695377631243, "instrumentalness": -7.041613633607729, "valence": -6.961253681056054}}, "energy__loudness__speechiness__valence": {"intercept": 47.59139327471303, "coef": {"energy": -8.200827389332991, "loudness": 0.768357319056012, "speechiness": -9.904925836652366, "valence": -4.789802980112903}}, "energy__instrumentalness__speechiness__valence": {"intercept": 37.6765386125663, "coef": {"energy": 1.932041772953546, "instrumentalness": -9.940664430088, "speechiness": -13.40813712361373, "valence": -7.074522187899079}}, "instrumentalness__loudness__speechiness__valence": {"intercept": 41.08261474049534, "coef": {"loudness": 0.27664815799808756, "instrumentalness": -8.448515974907078, "speechiness": -13.110794956414889, "valence": -7.426466042828309}}, "danceability__energy__instrumentalness__loudness__speechiness": {"intercept": 39.04196296572323, "coef": {"danceability": 4.794834255346955, "energy": -5.43095916985848, "loudness": 0.41270958384514156, "instrumentalness": -5.604151743031736, "speechiness": -12.629741842265144}}, "danceability__energy__instrumentalness__loudness__valence": {"intercept": 38.69211376353979, "coef": {"danceability": 10.80021330916903, "energy": -4.054975231166039, "loudness": 0.3876813241529957, "instrumentalness": -7.564792537008669, "valence": -10.43260141045296}}, "danceability__energy__loudness__speechiness__valence": {"intercept": 41.220079861020466, "coef": {"danceability": 10.719296155232929, "energy": -6.463173954318201, "loudness": 0.6554881189692727, "speechiness": -11.688825233246181, "valence": -8.065722109895944}}, "danceability__energy__instrumentalness__speechiness__valence": {"intercept": 32.17980358460315, "coef": {"danceability": 13.467730018265, "energy": 2.1008901797957993, "instrumentalness": -9.852849436049611, "speechiness": -15.33938677876802, "valence": -11.397927392237435}}, "danceability__instrumentalness__loudness__speechiness__valence": {"intercept": 35.53089544221186, "coef": {"danceability": 12.541090996983302, "loudness": 0.21620614610479472, "instrumentalness": -8.727880129381333, "speechiness": -14.809377209025008, "valence": -11.259107630961747}}, "energy__instrumentalness__loudness__speechiness__valence": {"intercept": 45.153346549373865, "coef": {"energy": -4.539764707218734, "loudness": 0.4674533644351132, "instrumentalness": -7.613870959688371, "speechiness": -11.99365276762032, "valence": -7.024805581956109}}, "danceability__energy__instrumentalness__loudness__speechiness__valence": {"intercept": 37.74892222519888, "coef": {"danceability": 12.085471140641024, "energy": -2.2486552658385324, "loudness": 0.3129124480169079, "instrumentalness": -8.30431092048241, "speechiness": -14.19431989495691, "valence": -10.920915003107712}}},
    testSamples: [{"danceability": 0.682, "energy": 0.822, "loudness": -8.477, "instrumentalness": 1.68e-05, "speechiness": 0.0524, "valence": 0.567, "popularity": 29.0}, {"danceability": 0.465, "energy": 0.0819, "loudness": -15.578, "instrumentalness": 0.0, "speechiness": 0.0448, "valence": 0.19, "popularity": 0.0}, {"danceability": 0.574, "energy": 0.935, "loudness": -2.174, "instrumentalness": 0.00974, "speechiness": 0.412, "valence": 0.322, "popularity": 22.0}, {"danceability": 0.754, "energy": 0.763, "loudness": -4.627, "instrumentalness": 2.23e-05, "speechiness": 0.0435, "valence": 0.464, "popularity": 0.0}, {"danceability": 0.363, "energy": 0.819, "loudness": -3.128, "instrumentalness": 0.0, "speechiness": 0.0634, "valence": 0.468, "popularity": 47.0}, {"danceability": 0.847, "energy": 0.875, "loudness": -6.454, "instrumentalness": 0.0966, "speechiness": 0.0659, "valence": 0.928, "popularity": 18.0}, {"danceability": 0.483, "energy": 0.789, "loudness": -9.402, "instrumentalness": 0.0277, "speechiness": 0.0341, "valence": 0.0736, "popularity": 11.0}, {"danceability": 0.42, "energy": 0.865, "loudness": -5.502, "instrumentalness": 0.0, "speechiness": 0.0365, "valence": 0.341, "popularity": 36.0}, {"danceability": 0.508, "energy": 0.891, "loudness": -7.492, "instrumentalness": 0.0, "speechiness": 0.0539, "valence": 0.572, "popularity": 63.0}, {"danceability": 0.51, "energy": 0.578, "loudness": -5.952, "instrumentalness": 0.0, "speechiness": 0.0276, "valence": 0.51, "popularity": 19.0}, {"danceability": 0.482, "energy": 0.769, "loudness": -9.992, "instrumentalness": 0.00102, "speechiness": 0.0304, "valence": 0.759, "popularity": 26.0}, {"danceability": 0.254, "energy": 0.989, "loudness": -4.192, "instrumentalness": 0.0, "speechiness": 0.473, "valence": 0.156, "popularity": 26.0}, {"danceability": 0.767, "energy": 0.659, "loudness": -7.429, "instrumentalness": 0.0, "speechiness": 0.0444, "valence": 0.439, "popularity": 75.0}, {"danceability": 0.891, "energy": 0.819, "loudness": -3.964, "instrumentalness": 0.000354, "speechiness": 0.106, "valence": 0.768, "popularity": 76.0}, {"danceability": 0.369, "energy": 0.637, "loudness": -6.279, "instrumentalness": 0.0, "speechiness": 0.0299, "valence": 0.121, "popularity": 36.0}, {"danceability": 0.649, "energy": 0.517, "loudness": -5.227, "instrumentalness": 0.0, "speechiness": 0.0273, "valence": 0.185, "popularity": 0.0}, {"danceability": 0.537, "energy": 0.307, "loudness": -10.967, "instrumentalness": 0.0, "speechiness": 0.0545, "valence": 0.114, "popularity": 66.0}, {"danceability": 0.686, "energy": 0.228, "loudness": -18.453, "instrumentalness": 0.937, "speechiness": 0.0494, "valence": 0.0968, "popularity": 50.0}, {"danceability": 0.499, "energy": 0.295, "loudness": -11.782, "instrumentalness": 0.0, "speechiness": 0.0287, "valence": 0.323, "popularity": 0.0}, {"danceability": 0.424, "energy": 0.079, "loudness": -16.861, "instrumentalness": 0.00147, "speechiness": 0.0376, "valence": 0.155, "popularity": 0.0}]
  }
};

// =========================================================================
// 2. STATE MANAGEMENT
// =========================================================================

let currentDatasetKey = "startup50";
let currentAlgo = "sfs";
let currentK = 2;

let rmseChartInstance = null;
let r2ChartInstance = null;
let scatterChartInstance = null;

// =========================================================================
// 3. INITIALIZATION
// =========================================================================

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  renderProblemView();
});

function onDatasetChange(newKey) {
  currentDatasetKey = newKey;
  currentAlgo = "sfs";
  const prob = PROBLEMS_DATA[currentDatasetKey];
  currentK = prob.defaultK;

  // Reset tab buttons visually
  document.querySelectorAll(".algo-tab-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById("tab-sfs").classList.add("active");

  renderProblemView();
}

// =========================================================================
// 4. CHART INITIALIZATION & UPDATE
// =========================================================================

function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(26, 32, 44, 0.95)',
        titleFont: { size: 13, weight: 'bold', family: 'Inter' },
        bodyFont: { size: 12, family: 'Inter' },
        padding: 10,
        cornerRadius: 8,
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(226, 232, 240, 0.6)' },
        ticks: { font: { family: 'Inter', weight: 600 } },
        title: {
          display: true,
          text: 'Number of Features (k)',
          font: { family: 'Inter', weight: 700, size: 11 },
          color: '#4a5568'
        }
      },
      y: {
        grid: { color: 'rgba(226, 232, 240, 0.6)' },
        ticks: { font: { family: 'Inter' } }
      }
    }
  };

  // RMSE Chart
  const ctxRmse = document.getElementById("rmseChart").getContext("2d");
  rmseChartInstance = new Chart(ctxRmse, {
    type: 'line',
    data: {
      labels: [1, 2, 3, 4, 5],
      datasets: [{
        label: 'RMSE',
        data: [],
        borderColor: '#1f77b4',
        backgroundColor: '#1f77b4',
        borderWidth: 2.2,
        pointRadius: 6,
        pointHoverRadius: 9,
        tension: 0.15
      }]
    },
    options: {
      ...commonOptions,
      onClick: (e, elements) => {
        if (elements.length > 0) {
          setK(elements[0].index + 1);
        }
      },
      plugins: {
        ...commonOptions.plugins,
        tooltip: {
          ...commonOptions.plugins.tooltip,
          callbacks: {
            label: (ctx) => `RMSE: ${ctx.parsed.y.toFixed(4)}`
          }
        }
      }
    }
  });

  // R-squared Chart
  const ctxR2 = document.getElementById("r2Chart").getContext("2d");
  r2ChartInstance = new Chart(ctxR2, {
    type: 'line',
    data: {
      labels: [1, 2, 3, 4, 5],
      datasets: [{
        label: 'R-squared',
        data: [],
        borderColor: '#1f77b4',
        backgroundColor: '#1f77b4',
        borderWidth: 2.2,
        pointRadius: 6,
        pointHoverRadius: 9,
        tension: 0.15
      }]
    },
    options: {
      ...commonOptions,
      onClick: (e, elements) => {
        if (elements.length > 0) {
          setK(elements[0].index + 1);
        }
      },
      plugins: {
        ...commonOptions.plugins,
        tooltip: {
          ...commonOptions.plugins.tooltip,
          callbacks: {
            label: (ctx) => `R²: ${ctx.parsed.y.toFixed(6)}`
          }
        }
      }
    }
  });

  // Scatter Chart
  const ctxScatter = document.getElementById("scatterChart").getContext("2d");
  scatterChartInstance = new Chart(ctxScatter, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Test Observations',
          data: [],
          backgroundColor: '#4338ca',
          borderColor: '#312e81',
          pointRadius: 7,
          pointHoverRadius: 10
        },
        {
          label: 'Ideal Prediction (y = x)',
          data: [],
          type: 'line',
          borderColor: '#e53e3e',
          borderDash: [5, 5],
          borderWidth: 2,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { font: { family: 'Inter', weight: 600 } }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `Actual: ${ctx.parsed.x.toFixed(1)} | Predicted: ${ctx.parsed.y.toFixed(1)}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(226, 232, 240, 0.6)' },
          title: { display: true, text: 'Actual Value', font: { family: 'Inter', weight: 700, size: 11 } }
        },
        y: {
          grid: { color: 'rgba(226, 232, 240, 0.6)' },
          title: { display: true, text: 'Predicted Value', font: { family: 'Inter', weight: 700, size: 11 } }
        }
      }
    }
  });
}

function updateCharts() {
  const prob = PROBLEMS_DATA[currentDatasetKey];
  const algo = prob.algorithms[currentAlgo];
  const records = algo.records;

  const ks = records.map(r => r.k);
  const rmseData = records.map(r => r.rmse);
  const r2Data = records.map(r => r.r2);

  const pointBgColors = records.map(r => {
    if (r.k === currentK) return '#e53e3e';
    if (r.k === algo.optimalK) return '#d97706';
    return '#1f77b4';
  });
  const pointRadii = records.map(r => (r.k === currentK ? 9 : 6));

  // Update RMSE Chart
  rmseChartInstance.data.labels = ks;
  rmseChartInstance.data.datasets[0].data = rmseData;
  rmseChartInstance.data.datasets[0].pointBackgroundColor = pointBgColors;
  rmseChartInstance.data.datasets[0].pointRadius = pointRadii;
  rmseChartInstance.update();

  // Update R2 Chart
  r2ChartInstance.data.labels = ks;
  r2ChartInstance.data.datasets[0].data = r2Data;
  r2ChartInstance.data.datasets[0].pointBackgroundColor = pointBgColors;
  r2ChartInstance.data.datasets[0].pointRadius = pointRadii;
  r2ChartInstance.update();

  // Compute Scatter Predictions
  const currentRecord = records.find(r => r.k === currentK) || records[0];
  const model = prob.models[currentRecord.modelKey];

  let scatterData = [];
  let minVal = 0;
  let maxVal = 100;

  if (currentDatasetKey === "startup50") {
    minVal = 60000;
    maxVal = 200000;
    scatterData = prob.testSamples.map(s => {
      let pred = model.intercept;
      if (model.coef["R&D Spend"]) pred += model.coef["R&D Spend"] * s.rd;
      if (model.coef["Administration"]) pred += model.coef["Administration"] * s.admin;
      if (model.coef["Marketing Spend"]) pred += model.coef["Marketing Spend"] * s.marketing;
      if (model.coef["State_New York"]) pred += model.coef["State_New York"] * s.state_ny;
      if (model.coef["State_Florida"]) pred += model.coef["State_Florida"] * s.state_fl;
      return { x: Math.round(s.profit), y: Math.round(pred) };
    });
  } else {
    // Spotify
    minVal = 0;
    maxVal = 100;
    scatterData = prob.testSamples.map(s => {
      let pred = model.intercept;
      for (const [feat, weight] of Object.entries(model.coef)) {
        if (s[feat] !== undefined) {
          pred += weight * s[feat];
        }
      }
      return { x: Math.round(s.popularity), y: Math.max(0, Math.min(100, Math.round(pred))) };
    });
  }

  scatterChartInstance.data.datasets[0].data = scatterData;
  scatterChartInstance.data.datasets[1].data = [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }];
  scatterChartInstance.options.scales.x.title.text = `Actual ${prob.targetName} (${prob.targetUnit})`;
  scatterChartInstance.options.scales.y.title.text = `Predicted ${prob.targetName} (${prob.targetUnit})`;
  scatterChartInstance.update();

  document.getElementById("model-status-badge").textContent = `${algo.name.split(' ')[0]} (k=${currentK})`;
}

// =========================================================================
// 5. VIEW CONTROLLER & PLAYGROUND GENERATOR
// =========================================================================

function renderProblemView() {
  const prob = PROBLEMS_DATA[currentDatasetKey];
  const algo = prob.algorithms[currentAlgo];

  // 1. Update Header
  document.getElementById("header-badge").textContent = prob.badge;
  document.getElementById("header-title").textContent = prob.title;
  document.getElementById("header-subtitle").textContent = prob.subtitle;

  // 2. Update Concepts Text
  if (currentDatasetKey === "startup50") {
    document.getElementById("concept-title-1").textContent = "Why Feature Selection?";
    document.getElementById("concept-body-1").innerHTML = "More features do not always mean a better model! In 50 Startups, administration overhead has almost no linear relationship with Profit ($r \approx 0.20$), and adding redundant features causes <strong>overfitting</strong>.";
    document.getElementById("scatter-title").textContent = "Actual vs. Predicted Profit (Validation Test Set)";
    document.getElementById("th-rmse").textContent = "RMSE ($)";
    document.getElementById("th-mae").textContent = "MAE ($)";
    document.getElementById("calc-title").textContent = "Live Startup Profit Prediction Playground";
    document.getElementById("calc-desc").textContent = "Adjust operational spending below. The active feature selection model will instantly compute the predicted annual profit!";
    document.getElementById("calc-result-label").textContent = "Predicted Annual Profit";
  } else {
    document.getElementById("concept-title-1").textContent = "Spotify Audio Analytics";
    document.getElementById("concept-body-1").innerHTML = "Spotify tracks comprise 20 audio and metadata dimensions. Feature selection pinpoints which audio metrics (e.g. <strong>danceability, loudness, speechiness, instrumentalness</strong>) carry true predictive signal for music popularity.";
    document.getElementById("scatter-title").textContent = "Actual vs. Predicted Popularity Score (Test Tracks)";
    document.getElementById("th-rmse").textContent = "RMSE (pts)";
    document.getElementById("th-mae").textContent = "MAE (pts)";
    document.getElementById("calc-title").textContent = "Live Spotify Song Popularity Prediction Playground";
    document.getElementById("calc-desc").textContent = "Fine-tune audio characteristics below. The active model will instantly predict the track's popularity score (0 to 100)!";
    document.getElementById("calc-result-label").textContent = "Predicted Popularity Score";
  }

  // 3. Update Slider Range
  const kSlider = document.getElementById("k-slider");
  kSlider.max = prob.maxK;
  if (currentK > prob.maxK) currentK = prob.maxK;
  kSlider.value = currentK;
  document.getElementById("k-value-display").textContent = `k = ${currentK}`;

  // 4. Update Playground Input Controls
  renderPlaygroundInputs();

  // 5. Update Main State & Charts
  renderState();
}

function renderPlaygroundInputs() {
  const container = document.getElementById("calc-inputs-container");
  container.innerHTML = "";

  if (currentDatasetKey === "startup50") {
    container.innerHTML = `
      <div class="parameter-control">
        <div class="control-label">
          <span>R&D Spend ($)</span>
          <span class="control-value" id="calc-rd-val">$100,000</span>
        </div>
        <input type="range" id="calc-rd-slider" min="0" max="200000" step="5000" value="100000" class="slider" oninput="updatePrediction()">
      </div>
      <div class="parameter-control">
        <div class="control-label">
          <span>Administration ($)</span>
          <span class="control-value" id="calc-admin-val">$120,000</span>
        </div>
        <input type="range" id="calc-admin-slider" min="0" max="200000" step="5000" value="120000" class="slider" oninput="updatePrediction()">
      </div>
      <div class="parameter-control">
        <div class="control-label">
          <span>Marketing Spend ($)</span>
          <span class="control-value" id="calc-mkt-val">$200,000</span>
        </div>
        <input type="range" id="calc-mkt-slider" min="0" max="500000" step="10000" value="200000" class="slider" oninput="updatePrediction()">
      </div>
      <div class="parameter-control">
        <div class="control-label">
          <span>Location / State</span>
        </div>
        <select id="calc-state-select" class="slider" style="height: 38px; border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0 10px; font-weight: 600;" onchange="updatePrediction()">
          <option value="California">California</option>
          <option value="Florida">Florida</option>
          <option value="New York">New York</option>
        </select>
      </div>
    `;
  } else {
    // Spotify Controls
    container.innerHTML = `
      <div class="parameter-control">
        <div class="control-label">
          <span>🕺 Danceability (0 - 1)</span>
          <span class="control-value" id="calc-dance-val">0.65</span>
        </div>
        <input type="range" id="calc-dance-slider" min="0" max="1" step="0.01" value="0.65" class="slider" oninput="updatePrediction()">
      </div>
      <div class="parameter-control">
        <div class="control-label">
          <span>⚡ Energy (0 - 1)</span>
          <span class="control-value" id="calc-energy-val">0.70</span>
        </div>
        <input type="range" id="calc-energy-slider" min="0" max="1" step="0.01" value="0.70" class="slider" oninput="updatePrediction()">
      </div>
      <div class="parameter-control">
        <div class="control-label">
          <span>🔊 Loudness (-30 to 0 dB)</span>
          <span class="control-value" id="calc-loudness-val">-6.5 dB</span>
        </div>
        <input type="range" id="calc-loudness-slider" min="-30" max="0" step="0.5" value="-6.5" class="slider" oninput="updatePrediction()">
      </div>
      <div class="parameter-control">
        <div class="control-label">
          <span>🎹 Instrumentalness (0 - 1)</span>
          <span class="control-value" id="calc-inst-val">0.05</span>
        </div>
        <input type="range" id="calc-inst-slider" min="0" max="1" step="0.01" value="0.05" class="slider" oninput="updatePrediction()">
      </div>
      <div class="parameter-control">
        <div class="control-label">
          <span>🗣️ Speechiness (0 - 1)</span>
          <span class="control-value" id="calc-speech-val">0.08</span>
        </div>
        <input type="range" id="calc-speech-slider" min="0" max="1" step="0.01" value="0.08" class="slider" oninput="updatePrediction()">
      </div>
      <div class="parameter-control">
        <div class="control-label">
          <span>😊 Valence / Mood (0 - 1)</span>
          <span class="control-value" id="calc-valence-val">0.55</span>
        </div>
        <input type="range" id="calc-valence-slider" min="0" max="1" step="0.01" value="0.55" class="slider" oninput="updatePrediction()">
      </div>
    `;
  }
}

function renderState() {
  const prob = PROBLEMS_DATA[currentDatasetKey];
  const algo = prob.algorithms[currentAlgo];
  const activeRecord = algo.records.find(r => r.k === currentK) || algo.records[0];
  const model = prob.models[activeRecord.modelKey];

  document.getElementById("current-algo-name").textContent = `${prob.name} — ${algo.name}`;
  document.getElementById("k-slider").value = currentK;
  document.getElementById("k-value-display").textContent = `k = ${currentK}`;

  // Included features chips
  const tagsContainer = document.getElementById("feature-tags-list");
  tagsContainer.innerHTML = "";
  activeRecord.features.forEach(feat => {
    const chip = document.createElement("span");
    chip.className = "feature-tag active";
    chip.innerHTML = `✓ ${feat}`;
    tagsContainer.appendChild(chip);
  });

  // Equation Display
  let eqStr = `${prob.targetName} = ${model.intercept.toFixed(2)}`;
  for (const [col, weight] of Object.entries(model.coef)) {
    const sign = weight >= 0 ? "+" : "−";
    eqStr += ` ${sign} (${Math.abs(weight).toFixed(4)} × ${col})`;
  }
  document.getElementById("equation-display").textContent = eqStr;

  // Metrics Grid
  const unitPrefix = currentDatasetKey === "startup50" ? "$" : "";
  const unitSuffix = currentDatasetKey === "startup50" ? "" : " pts";

  document.getElementById("metric-rmse").textContent = `${unitPrefix}${activeRecord.rmse.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 })}${unitSuffix}`;
  document.getElementById("metric-r2").textContent = activeRecord.r2.toFixed(4);
  document.getElementById("metric-mae").textContent = `${unitPrefix}${activeRecord.mae.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 })}${unitSuffix}`;

  const overfitBadge = document.getElementById("metric-overfit");
  if (currentK === algo.optimalK) {
    overfitBadge.textContent = "Optimal ★";
    overfitBadge.className = "metric-value highlight-good";
  } else if (currentK > algo.optimalK) {
    overfitBadge.textContent = "Overfitting ⚠";
    overfitBadge.className = "metric-value highlight-primary";
  } else {
    overfitBadge.textContent = "Underfitting";
    overfitBadge.className = "metric-value";
  }

  // Table
  renderTable();

  // Charts
  updateCharts();

  // Prediction Playground
  updatePrediction();
}

function renderTable() {
  const prob = PROBLEMS_DATA[currentDatasetKey];
  const algo = prob.algorithms[currentAlgo];
  const tbody = document.getElementById("table-body");
  tbody.innerHTML = "";

  algo.records.forEach(rec => {
    const tr = document.createElement("tr");
    if (rec.k === currentK) tr.classList.add("active-row");
    if (rec.k === algo.optimalK) tr.classList.add("optimal-row");

    tr.onclick = () => setK(rec.k);

    const isOptimal = rec.k === algo.optimalK;
    const optimalBadgeHtml = isOptimal ? `<span class="optimal-badge">★ Optimal</span>` : "";

    tr.innerHTML = `
      <td><strong>${rec.k}</strong> ${optimalBadgeHtml}</td>
      <td><code>${JSON.stringify(rec.features).replace(/"/g, "'")}</code></td>
      <td>${rec.rmse.toFixed(6)}</td>
      <td>${rec.mae.toFixed(6)}</td>
      <td><strong>${rec.r2.toFixed(6)}</strong></td>
    `;
    tbody.appendChild(tr);
  });
}

// =========================================================================
// 6. EVENT HANDLERS
// =========================================================================

function switchAlgorithm(algoKey) {
  currentAlgo = algoKey;
  document.querySelectorAll(".algo-tab-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById(`tab-${algoKey}`).classList.add("active");

  const prob = PROBLEMS_DATA[currentDatasetKey];
  currentK = prob.algorithms[algoKey].optimalK;

  renderState();
}

function onKSliderChange(val) {
  currentK = parseInt(val, 10);
  renderState();
}

function adjustK(delta) {
  const prob = PROBLEMS_DATA[currentDatasetKey];
  const nextK = Math.max(1, Math.min(prob.maxK, currentK + delta));
  if (nextK !== currentK) {
    currentK = nextK;
    renderState();
  }
}

function setK(k) {
  currentK = k;
  renderState();
}

function selectOptimalSubset() {
  const prob = PROBLEMS_DATA[currentDatasetKey];
  currentK = prob.algorithms[currentAlgo].optimalK;
  renderState();
}

function resetControls() {
  const prob = PROBLEMS_DATA[currentDatasetKey];
  currentAlgo = "sfs";
  currentK = prob.defaultK;

  document.querySelectorAll(".algo-tab-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById("tab-sfs").classList.add("active");

  renderPlaygroundInputs();
  renderState();
}

function updatePrediction() {
  const prob = PROBLEMS_DATA[currentDatasetKey];
  const algo = prob.algorithms[currentAlgo];
  const activeRecord = algo.records.find(r => r.k === currentK) || algo.records[0];
  const model = prob.models[activeRecord.modelKey];

  if (currentDatasetKey === "startup50") {
    const rd = parseFloat(document.getElementById("calc-rd-slider")?.value || 100000);
    const admin = parseFloat(document.getElementById("calc-admin-slider")?.value || 120000);
    const mkt = parseFloat(document.getElementById("calc-mkt-slider")?.value || 200000);
    const state = document.getElementById("calc-state-select")?.value || "California";

    if (document.getElementById("calc-rd-val")) document.getElementById("calc-rd-val").textContent = `$${rd.toLocaleString()}`;
    if (document.getElementById("calc-admin-val")) document.getElementById("calc-admin-val").textContent = `$${admin.toLocaleString()}`;
    if (document.getElementById("calc-mkt-val")) document.getElementById("calc-mkt-val").textContent = `$${mkt.toLocaleString()}`;

    let profit = model.intercept;
    if (model.coef["R&D Spend"]) profit += model.coef["R&D Spend"] * rd;
    if (model.coef["Administration"]) profit += model.coef["Administration"] * admin;
    if (model.coef["Marketing Spend"]) profit += model.coef["Marketing Spend"] * mkt;
    if (model.coef["State_New York"] && state === "New York") profit += model.coef["State_New York"];
    if (model.coef["State_Florida"] && state === "Florida") profit += model.coef["State_Florida"];

    document.getElementById("calc-predicted-profit").textContent = `$${profit.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    document.getElementById("calc-formula-desc").textContent = `Model: ${algo.name.split(' ')[0]} (k=${currentK}) using ${JSON.stringify(activeRecord.features).replace(/"/g, "'")}`;
  } else {
    // Spotify
    const dance = parseFloat(document.getElementById("calc-dance-slider")?.value || 0.65);
    const energy = parseFloat(document.getElementById("calc-energy-slider")?.value || 0.70);
    const loud = parseFloat(document.getElementById("calc-loudness-slider")?.value || -6.5);
    const inst = parseFloat(document.getElementById("calc-inst-slider")?.value || 0.05);
    const speech = parseFloat(document.getElementById("calc-speech-slider")?.value || 0.08);
    const val = parseFloat(document.getElementById("calc-valence-slider")?.value || 0.55);

    if (document.getElementById("calc-dance-val")) document.getElementById("calc-dance-val").textContent = dance.toFixed(2);
    if (document.getElementById("calc-energy-val")) document.getElementById("calc-energy-val").textContent = energy.toFixed(2);
    if (document.getElementById("calc-loudness-val")) document.getElementById("calc-loudness-val").textContent = `${loud.toFixed(1)} dB`;
    if (document.getElementById("calc-inst-val")) document.getElementById("calc-inst-val").textContent = inst.toFixed(2);
    if (document.getElementById("calc-speech-val")) document.getElementById("calc-speech-val").textContent = speech.toFixed(2);
    if (document.getElementById("calc-valence-val")) document.getElementById("calc-valence-val").textContent = val.toFixed(2);

    let pop = model.intercept;
    const inputs = {
      danceability: dance,
      energy: energy,
      loudness: loud,
      instrumentalness: inst,
      speechiness: speech,
      valence: val
    };

    for (const [col, weight] of Object.entries(model.coef)) {
      if (inputs[col] !== undefined) {
        pop += weight * inputs[col];
      }
    }

    pop = Math.max(0, Math.min(100, pop));
    document.getElementById("calc-predicted-profit").textContent = `${pop.toFixed(1)} / 100`;
    document.getElementById("calc-formula-desc").textContent = `Model: ${algo.name.split(' ')[0]} (k=${currentK}) using ${JSON.stringify(activeRecord.features).replace(/"/g, "'")}`;
  }
}

function exportResultsCSV() {
  const prob = PROBLEMS_DATA[currentDatasetKey];
  const algo = prob.algorithms[currentAlgo];
  let csvContent = "data:text/csv;charset=utf-8,Number of Features,Selected Features,RMSE,MAE,R-squared\n";

  algo.records.forEach(r => {
    const featStr = `"${r.features.join('; ')}"`;
    csvContent += `${r.k},${featStr},${r.rmse.toFixed(6)},${r.mae.toFixed(6)},${r.r2.toFixed(6)}\n`;
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `${currentDatasetKey}_${currentAlgo}_feature_selection.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
