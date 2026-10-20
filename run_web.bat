@echo off
chcp 65001 > nul
title FinTech AI Market Impact Forecaster - Streamlit
echo ==============================================================================
echo   HE THONG DU BAO TAC DONG CUA TIN TUC TAI CHINH DEN THI TRUONG
echo   Thuc nghiem so sanh: Bi-LSTM vs. BERT-base vs. FinBERT (Champion)
echo ==============================================================================
echo.
echo Dang khoi dong may chu Streamlit tren cong 8501...
echo.

cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501 --server.headless false

pause
