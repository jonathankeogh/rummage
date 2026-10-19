# Third-party notices

Parts of rummage are ported from DigUp (https://github.com/ARahim3/DigUp): the skip rules in `src/crawl.rs`, the
passage sizes in `src/extract.rs`, the embedding pipeline in `src/embed.rs`, the ranking in `src/search.rs`, and
`patches/llama.cpp-no-logits.patch`; and `tests/reference/` (its reference inputs and vectors, made with DigUp's
`scripts/testbed/make_reference_inputs.py` and llama.cpp b11461). DigUp is licensed as follows:

```
MIT License

Copyright (c) 2026 Abdur Rahim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

llama.cpp (vendor/llama.cpp) is MIT licensed by the ggml authors; see vendor/llama.cpp/LICENSE.

## Demo photos

The photos in `demo/photos` (shown in `docs/demo.gif`) come from Wikimedia Commons, released into the public
domain or under CC0 by their authors. Resized; the names are made up.

- `IMG_4412.jpg`: [Josie swims in Valois Bay.jpg](https://commons.wikimedia.org/wiki/File:Josie_swims_in_Valois_Bay.jpg), Carocaropilates, CC0
- `IMG_2207.jpg`: [Moraine Lake 17092005.jpg](https://commons.wikimedia.org/wiki/File:Moraine_Lake_17092005.jpg), Gorgo, Public domain
- `IMG_3518.jpg`: [Bike on a snowy street in Quebec City at night.jpg](https://commons.wikimedia.org/wiki/File:Bike_on_a_snowy_street_in_Quebec_City_at_night.jpg), Wilfredor, CC0
- `IMG_1093.jpg`: [Lighthouse Chania evening sun.jpg](https://commons.wikimedia.org/wiki/File:Lighthouse_Chania_evening_sun.jpg), Jebulon, CC0
- `IMG_5861.jpg`: [Cat Sleeping on Keyboard.jpg](https://commons.wikimedia.org/wiki/File:Cat_Sleeping_on_Keyboard.jpg), Panini!, CC0
- `IMG_6024.jpg`: [Birthday Cake Bloomsburg, Pennsylvania.jpg](https://commons.wikimedia.org/wiki/File:Birthday_Cake_Bloomsburg,_Pennsylvania.jpg), Tim-desser, CC0
- `cabin.jpg`: [A wood cabin in the snow (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:A_wood_cabin_in_the_snow_(Unsplash).jpg), Patrick Tomasso impatrickt, CC0
