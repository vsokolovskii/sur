#!/bin/bash
python data_preparation.py
cd src/audio && python svc.py --train && python svc.py --inference
cd ../image && python cnn.py --inference
cd .. && python main.py

