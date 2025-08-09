#!/bin/bash

rsync -avS --remove-source-files ./saves/ root@49.12.211.99:../mnt/volume/saves/
rsync -avS --remove-source-files ./logs/ root@49.12.211.99:../mnt/volume/logs/

echo done > /workspace/RLProject_StateEmbeddings/log.log
