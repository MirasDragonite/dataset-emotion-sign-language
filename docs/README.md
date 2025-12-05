Masters project by Miras Kabykenov to collect data based on emotion + sign


KSL data was taken from: https://special-edu.kz/https://special-edu.kz/news/6/single/1529


## command to run:
```
python tools/record_session.py   --root ./dataset   --split train   --participant participant_001   --glosses_file glosses.txt   --emotions neutral happy sad angry   --repeats 10   --fps 30 --cam 0   --countdown_sec 2.0   --record_sec 1.5
```
python tools/record_session.py   --root ./dataset   --split train   --participant participant_005   --glosses_file glosses.txt   --emotions neutral happy sad angry   --repeats 30   --fps 30 --cam 0   --countdown_sec 2.0   --record_sec 3


python tools/record_session.py   --root ./dataset   --split train   --participant participant_003   --glosses_file glosses.txt   --emotions neutral happy sad angry   --repeats 30   --fps 30 --cam 0   --countdown_sec 1.5   --record_sec 2.5
