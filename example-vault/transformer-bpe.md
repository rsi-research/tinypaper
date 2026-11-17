---
title: On one megabyte of text, the transformer is worse on byte-pair tokens than on characters
tinypaper: 0.5
date: 2026-11-15
---

Claims: ✅ 1 of 1 hold

Cites: [char-bigram](char-bigram.md), [neural-bigram](neural-bigram.md), [transformer](transformer.md), [bpe](bpe.md)

The first version of this paper predicted that tokens would help. They did not. The training
split is half as many tokens as characters, the model passes over it twice as often in the
same number of steps, and it memorises it. The claim now states what was found.

This paper defines nothing. It runs the transformer paper's model on the tokeniser paper's
data and compares it, per character, with the transformer paper's recorded result. Its two
parents both build on the bigram paper, and the comparison is only valid if they used the same
version of it.

```cite
name: base
at: ./char-bigram.md
```

```cite
name: nb
at: ./neural-bigram.md
```

```cite
name: tr
at: ./transformer.md
```

```cite
name: bpe
at: ./bpe.md
```

**Code `use`**

```python code=use
from base import bpc
from nb import train
```

**Experiments**

```python experiments
call("transformer_tok", train, d=bpe.tok256, make=tr.transformer, ctx=128, steps=3000, lr=1e-3, seeds=2, env=base.cpu)
```

**Claim `tokens_overfit`** (stated 2026-11-15): Per character, the same transformer on tokens is over 0.1 bits worse on test and over 0.2 bits better on train.

```python claim=tokens_overfit stated=2026-11-15
def tokens_overfit(r):
    """Per character, the same transformer on tokens is over 0.1 bits worse on test and over 0.2 bits better on train."""
    def gap(split):
        return bpc(r.transformer_tok, split) - bpc(r.tr.transformer, split)
    return gap("test") > 0.1 and gap("train") < -0.2
```

**Log**

```log
{"ran_on": {"python": "3.11.4", "device": "cuda", "packages": {"torch": "2.6.0"}, "gpu": "NVIDIA A40"}}
{"call":"86703f4a350eed98","fn":"fn:train","args":{"ctx":128,"d":"data:tok256@103ef11a","lr":0.001,"make":"fn:transformer","steps":3000},"seed":1,"seconds":48.66,"out":{"test_bpc":2.5327870845794678,"val_bpc":2.2769877910614014,"train_bpc":1.5601370334625244,"at_word_start":5.860908508300781,"elsewhere":4.485625743865967,"params":941377},"emitted":[{"step":0,"loss":8.583849476854159,"t":0.029},{"step":1,"loss":8.446799254104,"t":0.052},{"step":2,"loss":8.320369304009922,"t":0.075},{"step":3,"loss":8.213620360549982,"t":0.093},{"step":4,"loss":8.07445681498602,"t":0.103},{"step":5,"loss":8.014498159450417,"t":0.112},{"step":6,"loss":7.932055180071615,"t":0.133},{"step":7,"loss":7.862021780848469,"t":0.154},{"step":8,"loss":7.821940192168986,"t":0.173},{"step":9,"loss":7.786533779870558,"t":0.183},{"step":100,"loss":5.77124589257439,"t":1.561},{"step":200,"loss":5.353434680221301,"t":3.135},{"step":300,"loss":5.150068291263745,"t":4.831},{"step":400,"loss":4.71066293726832,"t":6.433},{"step":500,"loss":4.413642363887517,"t":8.029},{"step":600,"loss":4.360206322368537,"t":9.671},{"step":700,"loss":4.017236456226113,"t":11.343},{"step":800,"loss":4.00309535474157,"t":12.897},{"step":900,"loss":3.95548952508684,"t":14.462},{"step":1000,"loss":3.8672668967431414,"t":16.171},{"step":1100,"loss":3.8085052404575435,"t":17.828},{"step":1200,"loss":3.723834397819464,"t":19.367},{"step":1300,"loss":3.640560398271692,"t":20.923},{"step":1400,"loss":3.529026897464747,"t":22.669},{"step":1500,"loss":3.424777549852445,"t":24.291},{"step":1600,"loss":3.487773075036983,"t":25.811},{"step":1700,"loss":3.531898663768854,"t":27.536},{"step":1800,"loss":3.36210156843321,"t":29.199},{"step":1900,"loss":3.3806371706126,"t":30.734},{"step":2000,"loss":3.3152479908933534,"t":32.385},{"step":2100,"loss":3.2939180146027445,"t":33.993},{"step":2200,"loss":3.193159226872354,"t":35.633},{"step":2300,"loss":3.192606474632481,"t":37.186},{"step":2400,"loss":3.1815775711979506,"t":38.747},{"step":2500,"loss":3.237585441277933,"t":40.482},{"step":2600,"loss":3.145496298977537,"t":42.086},{"step":2700,"loss":2.9855844224481856,"t":43.721},{"step":2800,"loss":3.051480263056874,"t":45.363},{"step":2900,"loss":3.0679152691386258,"t":47.034}]}
{"call":"0782e55bd20b42f2","fn":"fn:train","args":{"ctx":128,"d":"data:tok256@103ef11a","lr":0.001,"make":"fn:transformer","steps":3000},"seed":2,"seconds":48.39,"out":{"test_bpc":2.4941468238830566,"val_bpc":2.2507729530334473,"train_bpc":1.5763384103775024,"at_word_start":5.87955379486084,"elsewhere":4.375309944152832,"params":941377},"emitted":[{"step":0,"loss":8.54763543402728,"t":0.027},{"step":1,"loss":8.417935061844394,"t":0.048},{"step":2,"loss":8.27394912275024,"t":0.069},{"step":3,"loss":8.14291416519666,"t":0.086},{"step":4,"loss":8.044423140700546,"t":0.095},{"step":5,"loss":7.92354479057632,"t":0.11},{"step":6,"loss":7.860938290148033,"t":0.131},{"step":7,"loss":7.793071184395155,"t":0.152},{"step":8,"loss":7.778823453667076,"t":0.165},{"step":9,"loss":7.758801921384231,"t":0.176},{"step":100,"loss":5.769894796869212,"t":1.617},{"step":200,"loss":5.351843496735517,"t":3.317},{"step":300,"loss":5.155564168854717,"t":4.951},{"step":400,"loss":4.906448330975843,"t":6.529},{"step":500,"loss":4.6167318607771834,"t":8.171},{"step":600,"loss":4.3491333913753865,"t":9.851},{"step":700,"loss":4.0667523252013265,"t":11.387},{"step":800,"loss":4.078045737952884,"t":12.926},{"step":900,"loss":4.048291707457713,"t":14.636},{"step":1000,"loss":3.8905003770135,"t":16.318},{"step":1100,"loss":3.870082940668369,"t":17.864},{"step":1200,"loss":3.8772074939630117,"t":19.445},{"step":1300,"loss":3.674716152733046,"t":21.158},{"step":1400,"loss":3.635817116760896,"t":22.765},{"step":1500,"loss":3.571761834481741,"t":24.326},{"step":1600,"loss":3.574698954193176,"t":25.925},{"step":1700,"loss":3.5337660513919222,"t":27.583},{"step":1800,"loss":3.5036367547525686,"t":29.081},{"step":1900,"loss":3.4297168915852247,"t":30.547},{"step":2000,"loss":3.414588265719427,"t":32.227},{"step":2100,"loss":3.347440735377759,"t":33.856},{"step":2200,"loss":3.394215888828822,"t":35.332},{"step":2300,"loss":3.3504562791779873,"t":36.904},{"step":2400,"loss":3.227895594763021,"t":38.75},{"step":2500,"loss":3.2721364118709695,"t":40.392},{"step":2600,"loss":3.2139395466108054,"t":41.846},{"step":2700,"loss":3.1768215629709906,"t":43.359},{"step":2800,"loss":3.1193346420935884,"t":45.072},{"step":2900,"loss":3.104357016995248,"t":46.73}]}
```
