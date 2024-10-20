import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian
from scipy.io import wavfile
def gen_graph(size:tuple,x,y,bits_range,
              title:str,line_lable,line_color:str,
              x_label:str,y_label:str,grid_enable:bool,file_name:str='polt.png'):
    plt.figure(figsize=size)
    plt.plot(x[bits_range], y[bits_range], label=line_lable, color=line_color)
    plt.title(title)  # 英文標題
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid_enable)
    plt.xlim([0, time_range])
    plt.legend()
    # plt.show()
    plt.savefig(f'./{file_name}')

# GMSK 調變的參數
Fs = 8000  # 取樣頻率 8kHz
bit_rate = 1000  # 位元率1kHz，模擬數位訊號
T = 1.0 / bit_rate  # 每個位元的時間
BT = [0.3]  # 高斯濾波器的 BT 係數（決定頻譜寬度）

# 使用者語音訊號經過ADC
Fs, audio_signal = wavfile.read("input.wav")

# 確保語音訊號範圍在 -1 到 1 之間，進行標準化處理
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# 定義時間軸，根據音訊信號長度和取樣頻率來設置時間
t = np.arange(0, len(audio_signal) / Fs, 1.0 / Fs)

# 生成二進制訊號，根據語音訊號生成二進制格式
audio_bits = (audio_signal > 0).astype(int)
audio_bits = 2 * audio_bits - 1  # 將訊號轉換為 -1 和 1（適合MSK相位轉換）

# 調整符號數量，使符號和時間軸匹配
num_symbols = len(t) // (Fs // bit_rate)  # 計算符號數量以匹配時間軸
symbols = np.repeat(audio_bits[:num_symbols], Fs // bit_rate)  # 展開符號至取樣頻率

for delta in range (0,501,500):
# 高斯濾波器設計
    for bt in BT:
        bt_product = bt * T  # B * T 的乘積決定頻譜限制
        gauss_filter = gaussian(Fs // bit_rate, std=bt_product * Fs)  # 使用高斯濾波器
        gauss_filter /= np.sum(gauss_filter)  # 濾波器歸一化

        # 應用高斯濾波器到符號流
        filtered_symbols = np.convolve(symbols, gauss_filter, mode='same')

        # 積分得到瞬時相位 (GMSK 調變)
        integrated_phase = np.cumsum(filtered_symbols) * (np.pi / 2) / (Fs / bit_rate)

        # 使用定義的時間軸 `t` 計算GMSK信號，符號數量匹配
        gmsk_signal = np.cos(2 * np.pi * (2000+delta) * t[:len(integrated_phase)] + integrated_phase)

        # 計算顯示500個位元的時間範圍
        time_range = 500 * T  # 500個位元的週期
        idx = np.where(t <= time_range)[0]  # 只取500個位元的索引
        # 繪製語音訊號
        gen_graph((10, 4),t,audio_signal,range(len(audio_signal)),
                f"BT={bt}\nUser Audio Signal (Time Domain)","Audio Signal",
                "blue","Time [s]","Amplitude",True,file_name=f"{bt}_{2000+delta}_audio_signal.png")

        # 計算顯示5個位元的時間範圍
        time_range = 5 * T  # 5個位元的週期
        idx = np.where(t <= time_range)[0]  # 只取5個位元的索引
        # 繪製高斯濾波後的訊號
        gen_graph((10, 4),t,filtered_symbols,range(len(filtered_symbols)),
                f"BT={bt}\nGaussian Filtered Data (Time Domain)","Filtered Symbols",
                "green","Time [s]","Amplitude",True,file_name=f"{bt}_{2000+delta}_filtered_symbols.png")

        # 計算顯示5個位元的時間範圍
        time_range = 5 * T  # 5個位元的週期
        idx = np.where(t <= time_range)[0]  # 只取5個位元的索引
        # 繪製GMSK調變後的訊號
        gen_graph((10, 4),t,gmsk_signal,range(len(gmsk_signal)),
                f"BT={bt}\nFR={2000+delta}\nGMSK Modulated Signal (Time Domain)","GMSK Modulated Signal",
                "red","Time [s]","Amplitude",True,file_name=f"{bt}_{2000+delta}_gmsk_signal.png")