import pandas as pd
import numpy as np

np.random.seed(42)

crops = {
    'rice':       dict(N=(60,100), P=(40,70), K=(40,70), temp=(20,35), hum=(75,90), pH=(5.5,7.0), rain=(150,250)),
    'maize':      dict(N=(70,120), P=(60,80), K=(50,80), temp=(18,30), hum=(55,75), pH=(5.8,7.5), rain=(55,100)),
    'chickpea':   dict(N=(20,50),  P=(60,90), K=(60,90), temp=(18,28), hum=(15,35), pH=(6.0,8.0), rain=(20,50)),
    'kidneybeans':dict(N=(15,40),  P=(50,80), K=(15,30), temp=(18,27), hum=(50,70), pH=(6.0,7.5), rain=(60,105)),
    'pigeonpeas': dict(N=(15,35),  P=(50,80), K=(15,35), temp=(20,30), hum=(40,70), pH=(6.0,7.5), rain=(40,90)),
    'mothbeans':  dict(N=(15,30),  P=(40,65), K=(15,30), temp=(25,35), hum=(40,65), pH=(5.5,7.5), rain=(30,65)),
    'mungbean':   dict(N=(15,30),  P=(40,70), K=(14,30), temp=(25,35), hum=(55,85), pH=(6.0,7.5), rain=(30,70)),
    'blackgram':  dict(N=(20,40),  P=(60,80), K=(17,30), temp=(25,35), hum=(60,80), pH=(6.0,7.5), rain=(50,100)),
    'lentil':     dict(N=(15,30),  P=(60,90), K=(20,30), temp=(15,25), hum=(50,70), pH=(6.0,7.5), rain=(30,60)),
    'pomegranate':dict(N=(15,20),  P=(10,25), K=(30,50), temp=(18,38), hum=(85,95), pH=(5.5,7.0), rain=(100,200)),
    'banana':     dict(N=(90,120), P=(60,80), K=(50,70), temp=(25,35), hum=(75,90), pH=(5.5,7.0), rain=(100,180)),
    'mango':      dict(N=(15,20),  P=(10,30), K=(30,50), temp=(24,35), hum=(45,65), pH=(5.5,7.5), rain=(50,110)),
    'grapes':     dict(N=(15,25),  P=(10,30), K=(20,40), temp=(8,42),  hum=(80,90), pH=(5.5,6.5), rain=(60,150)),
    'watermelon': dict(N=(80,120), P=(60,80), K=(50,80), temp=(25,35), hum=(70,90), pH=(6.0,7.0), rain=(55,80)),
    'muskmelon':  dict(N=(90,110), P=(60,80), K=(50,80), temp=(28,38), hum=(90,95), pH=(6.0,7.0), rain=(20,30)),
    'apple':      dict(N=(0,20),   P=(100,140),K=(190,220),temp=(0,25),hum=(90,95), pH=(5.5,6.5), rain=(95,115)),
    'orange':     dict(N=(0,20),   P=(5,25),  K=(5,25),  temp=(10,35), hum=(90,95), pH=(6.0,7.5), rain=(100,200)),
    'papaya':     dict(N=(40,65),  P=(50,80), K=(45,70), temp=(25,35), hum=(90,95), pH=(5.0,7.0), rain=(130,200)),
    'coconut':    dict(N=(0,20),   P=(0,25),  K=(25,55), temp=(25,35), hum=(85,95), pH=(5.0,8.0), rain=(120,200)),
    'cotton':     dict(N=(100,140),P=(30,55), K=(20,35), temp=(21,35), hum=(55,80), pH=(5.8,8.0), rain=(55,100)),
    'jute':       dict(N=(60,100), P=(40,70), K=(40,70), temp=(24,37), hum=(70,90), pH=(6.0,7.0), rain=(150,250)),
    'coffee':     dict(N=(90,120), P=(30,50), K=(30,50), temp=(15,30), hum=(50,70), pH=(6.0,7.0), rain=(140,250)),
}

rows = []
samples = 100
for crop, r in crops.items():
    for _ in range(samples):
        rows.append({
            'N':           np.random.uniform(*r['N']),
            'P':           np.random.uniform(*r['P']),
            'K':           np.random.uniform(*r['K']),
            'temperature': np.random.uniform(*r['temp']),
            'humidity':    np.random.uniform(*r['hum']),
            'ph':          np.random.uniform(*r['pH']),
            'rainfall':    np.random.uniform(*r['rain']),
            'label':       crop,
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('crop_data.csv', index=False)
print(f"Dataset created: {len(df)} rows, {df['label'].nunique()} crops")
