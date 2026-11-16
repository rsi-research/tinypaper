---
title: At equal size and tuned learning rates, the transformer's lead over the GRU is small
tinypaper: 0.3
date: 2026-11-15
---

The transformer paper claims a lead of over 0.05 bits per character over the GRU. But its GRU
has a seventh of the transformer's parameters and neither learning rate was tuned. This paper
reruns both at matched size, over three learning rates, picks each model's rate on the
validation split, and claims the lead mostly goes away. It challenges the transformer paper's
claim `beats_rnn`; it does not edit it.

```cite
name: base
at: ./char-bigram.md
```

```cite
name: nb
at: ./neural-bigram.md
```

```cite
name: rnn
at: ./rnn.md
```

```cite
name: tr
at: ./transformer.md
```

```python code=select
from base import bpc
from nb import train

RATES = [1e-3, 3e-3, 1e-2]


def tuned(runs):
    # the runs at the learning rate that is best on validation
    return runs.where(lr=min(RATES, key=lambda lr: bpc(runs.where(lr=lr), "val")))
```

```python experiments
for lr in RATES:
    run("gru", train, d=base.text, make=rnn.gru, width=370, ctx=128, steps=3000, lr=lr, seeds=2, env=base.cpu)
    run("transformer", train, d=base.text, make=tr.transformer, ctx=128, steps=3000, lr=lr, seeds=2, env=base.cpu)
```

```python claim=matched stated=2026-11-15 differs=make,width
def matched(r):
    # the two models are within 2% of each other in parameters
    a, b = r.gru[0].out["params"], r.transformer[0].out["params"]
    return abs(a - b) / b < 0.02
```

```python claim=gap_closes stated=2026-11-15 differs=make,width,lr challenges=tr.beats_rnn
def gap_closes(r):
    # with both tuned, the transformer leads by less than 0.05 bits per character
    return bpc(tuned(r.gru)) - bpc(tuned(r.transformer)) < 0.05
```

```results
{"set": "a40-torch2.6", "ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"run":"61fa29a083283c40","experiment":"gru","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.001,"make":"fn:gru","steps":3000,"width":370},"seed":1,"code":"39d47e4398abb5c7","seconds":35.23,"out":{"test_bpc":2.232825517654419,"val_bpc":2.0618324279785156,"train_bpc":1.7363711595535278,"params":871785},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[5.999742730441321,5.650121263294128,5.297424058418904,4.986383803305066,4.682120352565827,4.518723419599391,4.400147229239614,4.330547914227246,4.1681288746136165,4.093691343667175]},"prefix_t":0.124}
{"run":"3fb5e4281074547c","experiment":"gru","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.001,"make":"fn:gru","steps":3000,"width":370},"seed":2,"code":"39d47e4398abb5c7","seconds":35.28,"out":{"test_bpc":2.252286195755005,"val_bpc":2.0558249950408936,"train_bpc":1.7309560775756836,"params":871785},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.021581775378484,5.656671050569587,5.288425926125762,4.964239317179971,4.662037594459275,4.486630081087184,4.340492982996039,4.273005614936266,4.177772285812796,4.090376206089144]},"prefix_t":0.132}
{"run":"d040e7aa2baafe84","experiment":"transformer","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.001,"make":"fn:transformer","steps":3000},"seed":1,"code":"39d47e4398abb5c7","seconds":43.51,"out":{"test_bpc":2.3091225624084473,"val_bpc":2.095316171646118,"train_bpc":1.852841854095459,"params":875585},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.312497996549563,5.848710318176017,5.476633420270968,5.150007409405339,4.926060544549634,4.8014333166019005,4.756172642374335,4.593830994953594,4.58846823193439,4.503627470437258]},"prefix_t":0.179}
{"run":"58044ec27e7560e4","experiment":"transformer","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.001,"make":"fn:transformer","steps":3000},"seed":2,"code":"39d47e4398abb5c7","seconds":43.13,"out":{"test_bpc":2.3415350914001465,"val_bpc":2.1049256324768066,"train_bpc":1.888541340827942,"params":875585},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.271956182296371,5.731367931353529,5.299441758878826,5.006905805101922,4.8430799474042345,4.697117239721064,4.660617705693752,4.676487576784768,4.612495928086433,4.5370760322382635]},"prefix_t":0.16}
{"run":"eb01c337f472f9a6","experiment":"gru","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.003,"make":"fn:gru","steps":3000,"width":370},"seed":1,"code":"39d47e4398abb5c7","seconds":35.43,"out":{"test_bpc":2.289978265762329,"val_bpc":2.092442512512207,"train_bpc":1.7695986032485962,"params":871785},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[5.999742730441321,5.107882322868305,4.468556768273314,4.26564131782632,4.019038146476553,3.8877799554421517,3.8189353003017694,3.7663533246269743,3.6041117711090354,3.5639514143754574]},"prefix_t":0.151}
{"run":"03cd4f167d03e3e9","experiment":"gru","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.003,"make":"fn:gru","steps":3000,"width":370},"seed":2,"code":"39d47e4398abb5c7","seconds":35.28,"out":{"test_bpc":2.297980308532715,"val_bpc":2.0904502868652344,"train_bpc":1.7621278762817383,"params":871785},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.021581775378484,5.077596522016665,4.530953105902245,4.295949132457269,4.079573975788451,3.9146549963618793,3.7815559030326433,3.7460074330646025,3.703682846652567,3.613632042730197]},"prefix_t":0.114}
{"run":"e2795e56acd2f049","experiment":"transformer","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.003,"make":"fn:transformer","steps":3000},"seed":1,"code":"39d47e4398abb5c7","seconds":43.05,"out":{"test_bpc":2.3591737747192383,"val_bpc":2.1022138595581055,"train_bpc":1.7121162414550781,"params":875585},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.312497996549563,5.270611275218789,4.8126355785485,4.61032481910194,4.479584295846637,4.386499373997745,4.307838981042019,4.176111965301366,4.149444679423881,4.061407792343114]},"prefix_t":0.165}
{"run":"be6f9d335574dbc6","experiment":"transformer","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.003,"make":"fn:transformer","steps":3000},"seed":2,"code":"39d47e4398abb5c7","seconds":43.29,"out":{"test_bpc":2.370102643966675,"val_bpc":2.1019973754882812,"train_bpc":1.7388864755630493,"params":875585},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.271956182296371,5.105783102631873,4.698574620704475,4.6229593525649255,4.4595245834153,4.312461186766607,4.25865297479116,4.267695134642924,4.15723033402844,4.089021670730949]},"prefix_t":0.198}
{"run":"622299177eaad20c","experiment":"gru","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.01,"make":"fn:gru","steps":3000,"width":370},"seed":1,"code":"39d47e4398abb5c7","seconds":35.21,"out":{"test_bpc":2.6068108081817627,"val_bpc":2.429732084274292,"train_bpc":2.2688894271850586,"params":871785},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[5.999742730441321,4.5828492147654,4.851099154448666,4.531029466199229,3.89596804944973,3.8039009209286445,3.669800544606085,3.5982922221691394,3.463827242626751,3.4013607360762665]},"prefix_t":0.141}
{"run":"3649cbb4c9487cb8","experiment":"gru","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.01,"make":"fn:gru","steps":3000,"width":370},"seed":2,"code":"39d47e4398abb5c7","seconds":35.06,"out":{"test_bpc":2.5052945613861084,"val_bpc":2.3296215534210205,"train_bpc":2.1549434661865234,"params":871785},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.021581775378484,4.737638416056152,4.765343445248943,4.553562289149972,4.054784740458388,3.702406735383165,3.554825670955375,3.5818314186897586,3.4692278418291447,3.3983187069478054]},"prefix_t":0.13}
{"run":"73867f4b1e753453","experiment":"transformer","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.01,"make":"fn:transformer","steps":3000},"seed":1,"code":"39d47e4398abb5c7","seconds":43.45,"out":{"test_bpc":2.394659996032715,"val_bpc":2.1076014041900635,"train_bpc":1.7569222450256348,"params":875585},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.312497996549563,4.915518695982348,4.9861093189942896,5.379104810689083,4.761529214018108,4.532467585125744,4.398661643101462,4.268951639890127,4.212026414350443,4.102788194001913]},"prefix_t":0.143}
{"run":"ea1bd6647f1e784e","experiment":"transformer","fn":"fn:train","args":{"ctx":128,"d":"data:char-bigram/text","lr":0.01,"make":"fn:transformer","steps":3000},"seed":2,"code":"39d47e4398abb5c7","seconds":43.2,"out":{"test_bpc":2.3651416301727295,"val_bpc":2.117946147918701,"train_bpc":1.8046587705612183,"params":875585},"logged":["test_bits"],"prefix":{"step":[0,1,2,3,4,5,6,7,8,9],"loss":[6.271956182296371,4.873053083796605,5.428607952948132,4.705519280146316,6.1912337173568,4.542932041500142,4.473241575682817,4.492721018650141,4.423841623014285,4.319577484894009]},"prefix_t":0.178}
```
