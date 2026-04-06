# Todo:

* ✅ zie of ik de performance van run_onnx_v2 kan fixen. eerst maar zien waar de tijd precies in zit en dan [OPTIMIZEN].
* ✅ propere crate maken, inclusief mask enzo, met mooie visualisatie examples, en from_hf ding en bon builder enzo
* ✅ serde feature voor crate
* ✅ thiserror error handling
* ✅ execution providers toevoegen
* ✅ kleinere models pullen en kijken of die ook werken
* ✅ kan ik t ook werkend krijgen met class list als model input? Dus je zegt egg als input en dan detect ie alleen eggs
* ✅ ik wil graag text-prompt supporten, het liefst in rust only, kan dit?
    * ✅ maak eerst in python only
    * ✅ dan export onnx ervoor maken
    * ✅ dan in python onnx uitvoeren met ultralytics pre&post processing zien of de results matchen <- we are here, de
      results matchen niet
    * ✅ dan in python onnx uitvoeren zien of de results matchen
    * ✅ dan porten naar rust
        * ✅ de promptable_basic.rs example werkt soortvan, niet exact zelfde results als python onnx impl
        * ✅ maak mapje in src die promptable heet ofzo, en dan daarin de code zetten om 'm te runnen
        * ✅ maak example die visualized met mask+bbox+tags+score
        * ✅ zie of ik 'm gelijk kan krijgen met python-onnx
* ✅ make constructor basically yolo only because it doesnt support anything else anyways
    * ✅ arguments: scale (n, s, m, l, x)
    * ✅ include_mask: (seg vs det model) (todo: maak det models voor promptable models)
    * ✅ 2 verschillende structs denk ik, 1 voor promptable, 1 voor prompt free
* ✅ join de 2 export_onnx scripts
* ✅ mutex in inner models of in de wrapper struct
* ✅ maak benchmark met from_hf (is nu toch async)
* ✅ cache embeddings in promptable detector
* ✅ test in CI (requires from_hf)
* benchmark alle model sizes en laat speed zien in readme
    * 26n: 169ms
* test of _det_ voor promptable wel sneller is
* leg in readme uit hoe je een onnx export in python
* uv script voor export_onnx package versies pinnen (ook git clip)
* video support/helpers?
* maak promptable een feature, alleen open_clip_embedder binnenhalen als ie enabled is
* haal mut self uit objectdetector
* regression test broke at some point?


benchmark results:
(promptable is with cached embeddings)

full_predict/prompt_free/nano/seg
time:   [175.31 ms 178.02 ms 180.94 ms]
full_predict/prompt_free/nano/det
time:   [131.92 ms 137.54 ms 143.86 ms]
full_predict/promptable/nano/seg
time:   [53.725 ms 55.733 ms 57.863 ms]
full_predict/promptable/nano/det
time:   [39.873 ms 41.508 ms 43.306 ms]
full_predict/prompt_free/small/seg
time:   [133.13 ms 135.53 ms 138.24 ms]
full_predict/prompt_free/small/det
time:   [72.932 ms 75.296 ms 77.900 ms]
full_predict/promptable/small/seg
time:   [83.623 ms 86.745 ms 90.183 ms]
full_predict/promptable/small/det
time:   [55.710 ms 58.095 ms 60.747 ms]
full_predict/prompt_free/medium/seg
time:   [233.91 ms 242.42 ms 252.07 ms]
full_predict/prompt_free/medium/det
time:   [122.83 ms 126.02 ms 129.52 ms]
full_predict/promptable/medium/seg
time:   [161.80 ms 165.66 ms 169.92 ms]
full_predict/promptable/medium/det
time:   [106.18 ms 109.37 ms 112.70 ms]
full_predict/prompt_free/large/seg
time:   [236.86 ms 241.22 ms 245.93 ms]
full_predict/prompt_free/large/det
time:   [184.64 ms 195.00 ms 205.75 ms]
full_predict/promptable/large/seg
time:   [183.45 ms 187.57 ms 191.85 ms]
full_predict/promptable/large/det
time:   [127.93 ms 132.41 ms 137.37 ms]
full_predict/prompt_free/xlarge/seg
time:   [407.25 ms 422.37 ms 439.75 ms]
full_predict/prompt_free/xlarge/det
time:   [342.08 ms 358.87 ms 376.05 ms]
full_predict/promptable/xlarge/seg
time:   [386.06 ms 401.46 ms 417.96 ms]
full_predict/promptable/xlarge/det
time:   [236.90 ms 242.91 ms 249.31 ms]