import joblib
import gradio as gr
import cv2
import xgboost as xgb
import joblib
import pandas as pd
from ultralytics import YOLO


xgb_model = joblib.load('xgboost_model.pkl')    