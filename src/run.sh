#!/bin/bash
cd audio && python svc.py --inference
cd ../image && python cnn.py --inference
cd .. && python main.py

