# Data

Videos are not committed (size and licensing). Download them by hand into
`data/videos/` (Wikimedia rate-limits scripted downloads) and check the SHA-256.

| File | Source | Author | License | Use |
|---|---|---|---|---|
| `Motorway_A40_-_on_bridge_above_the_traffic.webm` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Motorway_A40_-_on_bridge_above_the_traffic.webm) | Kathinka Engels | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | **Main video.** Fixed camera on a bridge, dense mixed traffic (cars, trucks). 25 s, 1280×720, 30 fps |
| `E18_nord_for_Ramsum,_Vestfold.webm` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:E18_nord_for_Ramsum,_Vestfold.webm) | PeterFisk | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Second video, light traffic. Camera pans slowly for the first ~25 s, stable from 25 s to the end (48 s). 1920×1080, 30 fps |
| `Lake_Shore_Drive_Traffic.webm` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Lake_Shore_Drive_Traffic.webm) | Sea Cow | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Failure case only: high-altitude drone, cars 8–20 px seen from above. At imgsz 1920, 98% of detections are parked cars, 74% of frames find no moving vehicle |
| `Jane_M._Byrne_Interchange_Traffic.webm` | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Jane_M._Byrne_Interchange_Traffic.webm) | Sea Cow | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Not used: drone rotates during the clip |

SHA-256:

```
f4c7c1d826a166b78353386ecdc10ed2090255eaca832c083a64f8ea633257dc  Motorway_A40_-_on_bridge_above_the_traffic.webm
adc18f505d9b7d00373775c8dd80315e38775aa1033d23c106b9918487bafb78  E18_nord_for_Ramsum,_Vestfold.webm
9420a0bbad132a2e8cf060e10a47c336b50f9a4ab7ba3d8ce42aa5ec0fad753f  Lake_Shore_Drive_Traffic.webm
f9cc9b37426aa60bc4f12f1fcfd64c577d805543cf8be45e3fcd7795f6bce3a1  Jane_M._Byrne_Interchange_Traffic.webm
```

Annotated outputs derived from these videos keep the source license and credit
the author (CC BY-SA sources: same license).
