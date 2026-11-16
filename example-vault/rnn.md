---
title: A recurrent network beats a fixed window of characters
tinypaper: 0.4
date: 2026-11-15
---

Cites: [char-bigram](char-bigram.md), [neural-bigram](neural-bigram.md), [mlp-lm](mlp-lm.md)

A GRU reads the whole window instead of the last eight characters. Everything else is cited.

```cite
name: base
at: ./char-bigram.md
```

```cite
name: nb
at: ./neural-bigram.md
```

```cite
name: mlp
at: ./mlp-lm.md
```

```python code=model
import torch
from base import bpc
from nb import train


class GRU(torch.nn.Module):
    def __init__(self, vocab, width=128, layers=1):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, width)
        self.rnn = torch.nn.GRU(width, width, layers, batch_first=True)
        self.out = torch.nn.Linear(width, vocab)

    def forward(self, x):
        return self.out(self.rnn(self.embed(x))[0])


def gru(vocab, **kw):
    return GRU(vocab, **kw)
```

```python experiments
run("gru", train, d=base.text, make=gru, ctx=128, steps=3000, seeds=2, env=base.cpu)
```

```python claim=beats_window stated=2026-11-15 differs=make,k,ctx
def beats_window(r):
    # the GRU beats the 8-character MLP by more than 0.1 bits per character
    return bpc(r.mlp.mlp.where(k=8)) - bpc(r.gru) > 0.1
```

```log
{"log": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}, "history": {"runs": 2, "claims_edited": []}}
{"call":"b0273cb0999c313e","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","make":"fn:gru","steps":3000},"seed":1,"seconds":12.18,"out":{"test_bpc":2.391141176223755,"val_bpc":2.2012479305267334,"train_bpc":2.0160043239593506,"at_word_start":4.021728038787842,"elsewhere":2.01400089263916,"params":115777},"emitted":[{"step":0,"loss":6.035894171583287,"t":0.065},{"step":1,"loss":5.772923067385604,"t":0.07},{"step":2,"loss":5.498965711450805,"t":0.076},{"step":3,"loss":5.168386507372444,"t":0.081},{"step":4,"loss":4.852627048318931,"t":0.084},{"step":5,"loss":4.596697601778175,"t":0.086},{"step":6,"loss":4.5469402691611505,"t":0.089},{"step":7,"loss":4.45621770100451,"t":0.091},{"step":8,"loss":4.402875905978203,"t":0.093},{"step":9,"loss":4.2882484372822365,"t":0.096},{"step":100,"loss":2.8587114450454845,"t":0.428},{"step":200,"loss":2.547542718891452,"t":0.846},{"step":300,"loss":2.3988229573322997,"t":1.274},{"step":400,"loss":2.3331646247644526,"t":1.697},{"step":500,"loss":2.2540957730129274,"t":2.0},{"step":600,"loss":2.212042059276001,"t":2.431},{"step":700,"loss":2.206131531513799,"t":2.854},{"step":800,"loss":2.100351366326167,"t":3.278},{"step":900,"loss":2.1514186907948134,"t":3.707},{"step":1000,"loss":2.1486103860888743,"t":4.094},{"step":1100,"loss":2.1219764648456567,"t":4.462},{"step":1200,"loss":2.07143729909781,"t":4.871},{"step":1300,"loss":2.126987523343846,"t":5.294},{"step":1400,"loss":2.142029297970958,"t":5.715},{"step":1500,"loss":2.0599509218119847,"t":6.043},{"step":1600,"loss":2.0717559829498584,"t":6.429},{"step":1700,"loss":2.0519115927974028,"t":6.858},{"step":1800,"loss":2.068643956882496,"t":7.285},{"step":1900,"loss":2.040168445413839,"t":7.719},{"step":2000,"loss":2.037769287434303,"t":8.122},{"step":2100,"loss":2.036112406575891,"t":8.493},{"step":2200,"loss":1.9837249432792214,"t":8.883},{"step":2300,"loss":2.0546867048517248,"t":9.298},{"step":2400,"loss":2.0893281383191153,"t":9.711},{"step":2500,"loss":1.9756063302625917,"t":10.127},{"step":2600,"loss":1.9511816981523573,"t":10.437},{"step":2700,"loss":2.0114179336864524,"t":10.865},{"step":2800,"loss":2.0300924978477397,"t":11.28},{"step":2900,"loss":1.9702447711219435,"t":11.7}]}
{"call":"edbc2592b6dddc49","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","make":"fn:gru","steps":3000},"seed":2,"seconds":11.98,"out":{"test_bpc":2.3624839782714844,"val_bpc":2.1879334449768066,"train_bpc":2.01430082321167,"at_word_start":4.029294490814209,"elsewhere":1.9769651889801025,"params":115777},"emitted":[{"step":0,"loss":6.06049181824029,"t":0.016},{"step":1,"loss":5.809864940790936,"t":0.021},{"step":2,"loss":5.575107246501995,"t":0.026},{"step":3,"loss":5.280389520816245,"t":0.03},{"step":4,"loss":4.970083631621591,"t":0.033},{"step":5,"loss":4.724677459521828,"t":0.036},{"step":6,"loss":4.495515048796059,"t":0.039},{"step":7,"loss":4.45624556219395,"t":0.041},{"step":8,"loss":4.476378883199856,"t":0.044},{"step":9,"loss":4.358437308821762,"t":0.049},{"step":100,"loss":2.8764999545895265,"t":0.407},{"step":200,"loss":2.556614287776513,"t":0.795},{"step":300,"loss":2.4148030693025193,"t":1.212},{"step":400,"loss":2.3722119097816465,"t":1.611},{"step":500,"loss":2.2630276919854873,"t":1.967},{"step":600,"loss":2.168770020440362,"t":2.348},{"step":700,"loss":2.2172435025686954,"t":2.755},{"step":800,"loss":2.1535132674996715,"t":3.186},{"step":900,"loss":2.092727203430768,"t":3.607},{"step":1000,"loss":2.125676327613668,"t":4.015},{"step":1100,"loss":2.0891024970811833,"t":4.408},{"step":1200,"loss":2.0698672694780833,"t":4.78},{"step":1300,"loss":2.095020936045325,"t":5.177},{"step":1400,"loss":2.070293958434874,"t":5.58},{"step":1500,"loss":2.0888354080243934,"t":6.008},{"step":1600,"loss":2.0357914869493805,"t":6.397},{"step":1700,"loss":2.0331553368769555,"t":6.719},{"step":1800,"loss":2.0444845220198924,"t":7.132},{"step":1900,"loss":2.0973132927986753,"t":7.546},{"step":2000,"loss":2.023980922366679,"t":7.983},{"step":2100,"loss":1.9984158732985733,"t":8.406},{"step":2200,"loss":2.0143767232118965,"t":8.792},{"step":2300,"loss":2.0035368287106645,"t":9.173},{"step":2400,"loss":2.0273584896469736,"t":9.58},{"step":2500,"loss":2.006073228845589,"t":10.003},{"step":2600,"loss":2.0215717893933927,"t":10.351},{"step":2700,"loss":2.0330186106695196,"t":10.727},{"step":2800,"loss":1.977679237153441,"t":11.101},{"step":2900,"loss":1.9817177337610015,"t":11.505}]}
```
