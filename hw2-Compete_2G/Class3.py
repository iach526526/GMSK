import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, lfilter
import binascii


# 低通濾波器設計
def butter_lowpass_filter(data, cutoff, Fs, order=5):
    nyq = 0.5 * Fs  # 奈奎斯特頻率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


# 語音量化
def quantize_audio(signal):
    quantized_signal = np.round(signal * 32767).astype(np.int16)
    return quantized_signal


# 語音反量化
def dequantize_audio(quantized_signal):
    dequantized_signal = quantized_signal.astype(np.float32) / 32767
    return dequantized_signal


# FSK 調變
def fsk_modulate(bits, bit_rate=1000, f0=1000, f1=2000, Fs=8000):
    samples_per_bit = Fs // bit_rate
    t = np.arange(len(bits) * samples_per_bit) / Fs
    fsk_signal = np.zeros_like(t)

    for i, bit in enumerate(bits):
        freq = f1 if bit == 1 else f0
        fsk_signal[i * samples_per_bit:(i + 1) * samples_per_bit] = np.cos(
            2 * np.pi * freq * t[i * samples_per_bit:(i + 1) * samples_per_bit])

    return fsk_signal, t


# FSK 解調
def fsk_demodulate(fsk_signal, bit_rate=1000, f0=1000, f1=2000, Fs=8000):
    samples_per_bit = Fs // bit_rate
    t = np.arange(len(fsk_signal)) / Fs

    ref_wave_0 = np.cos(2 * np.pi * f0 * t)
    ref_wave_1 = np.cos(2 * np.pi * f1 * t)

    demodulated_bits = np.zeros(len(fsk_signal) // samples_per_bit, dtype=int)

    for i in range(len(demodulated_bits)):
        bit_segment = fsk_signal[i * samples_per_bit:(i + 1) * samples_per_bit]
        match_0 = np.sum(bit_segment * ref_wave_0[i * samples_per_bit:(i + 1) * samples_per_bit])
        match_1 = np.sum(bit_segment * ref_wave_1[i * samples_per_bit:(i + 1) * samples_per_bit])
        demodulated_bits[i] = 1 if match_1 > match_0 else 0

    return demodulated_bits


# CRC16 校驗碼生成
def generate_crc(data_bits):
    data_bytes = np.packbits(data_bits).tobytes()
    crc_value = binascii.crc_hqx(data_bytes, 0xFFFF)  # 使用CRC-16校驗
    crc_bits = np.unpackbits(np.array([crc_value >> 8, crc_value & 0xFF], dtype=np.uint8))
    print(f"Generated CRC: 0x{crc_value:04x}")
    return np.concatenate([data_bits, crc_bits])


# CRC16 校驗驗證
def check_crc(data_bits_with_crc):
    data_bits = data_bits_with_crc[:-16]  # 提取原始數據
    crc_received = data_bits_with_crc[-16:]  # 提取CRC校驗碼
    expected_crc = generate_crc(data_bits)[-16:]  # 重新生成校驗碼
    return np.array_equal(crc_received, expected_crc)


# XOR加密
def xor_encrypt(bits, key=123):
    key_bits = np.unpackbits(np.array([key], dtype=np.uint8))
    encrypted_bits = np.bitwise_xor(bits, np.tile(key_bits, len(bits) // len(key_bits) + 1)[:len(bits)])
    return encrypted_bits


# XOR解密
def xor_decrypt(bits, key=123):
    return xor_encrypt(bits, key)  # XOR加密與解密是相同的操作


# 交錯，加入填充
def interleave(bits, block_size=10):
    pad_size = (block_size - len(bits) % block_size) % block_size  # 計算需要填充的位數
    padded_bits = np.hstack([bits, np.zeros(pad_size, dtype=bits.dtype)])  # 填充位元
    interleaved_bits = padded_bits.reshape((-1, block_size)).T.flatten()  # 交錯
    return interleaved_bits, pad_size


# 解交錯並移除填充
def deinterleave(bits, pad_size, block_size=10):
    deinterleaved_bits = bits.reshape((block_size, -1)).T.flatten()  # 解交錯
    if pad_size > 0:
        deinterleaved_bits = deinterleaved_bits[:-pad_size]  # 移除填充的部分
    return deinterleaved_bits


# 模擬基地台上下行傳輸中加入干擾
def add_noise(signal, noise_level=0.1):  # 增加噪聲強度
    noise = np.random.normal(0, noise_level, len(signal))
    return signal + noise


# 模擬 A 手機至 B 手機的完整流程
def simulate_fsk_transmission(audio_signal, Fs=8000, noise=False, noise_level=0.1):
    cutoff_freq = 3500  # 低通濾波器的截止頻率
    # 進行低通濾波，減少高頻噪音
    audio_signal_filtered = butter_lowpass_filter(audio_signal, cutoff_freq, Fs)

    # 量化音訊信號
    quantized_audio = quantize_audio(audio_signal_filtered)
    encoded_bits = np.unpackbits(quantized_audio.view(np.uint8))

    # 加密
    encrypted_bits = xor_encrypt(encoded_bits)
    print("Data encrypted.")

    # 交錯，並處理填充
    interleaved_bits, pad_size = interleave(encrypted_bits)
    print("Data interleaved.")

    # 生成並印出 CRC 校驗碼
    encoded_bits_crc = generate_crc(interleaved_bits)
    print("Generated CRC and appended to data.")

    # FSK 調變
    fsk_signal, time = fsk_modulate(encoded_bits_crc)

    # 模擬基地台上下行傳輸過程
    fsk_signal_with_noise = fsk_signal
    if noise:
        fsk_signal_with_noise = add_noise(fsk_signal, noise_level)  # 加入噪聲
        print("Noise added to the signal.")

    # 接收端解調
    demodulated_bits = fsk_demodulate(fsk_signal_with_noise)

    # 驗證 CRC 校驗碼並印出結果
    try:
        crc_check_passed = check_crc(demodulated_bits)
        print("CRC check passed:", crc_check_passed)

        if crc_check_passed:
            decoded_bits = demodulated_bits[:-16]  # 去除CRC位元

            # 解交錯並移除填充
            deinterleaved_bits = deinterleave(decoded_bits, pad_size)
            print("Data deinterleaved.")

            # 解密
            decrypted_bits = xor_decrypt(deinterleaved_bits)
            print("Data decrypted.")

            decoded_audio = np.packbits(decrypted_bits[:len(encoded_bits)]).view(np.int16)  # 確保數據長度一致
            restored_audio_signal = dequantize_audio(decoded_audio)

            # 檢查還原音訊振幅
            print(f"Restored audio signal max: {np.max(restored_audio_signal)}")
            print(f"Restored audio signal min: {np.min(restored_audio_signal)}")

            # 再次進行低通濾波，過濾掉高頻噪音
            restored_audio_signal_filtered = butter_lowpass_filter(restored_audio_signal, cutoff_freq, Fs)

            # 放大信號，確保還原的音訊足夠大
            restored_audio_signal_filtered = restored_audio_signal_filtered * 10

            return restored_audio_signal_filtered, restored_audio_signal, time, fsk_signal, fsk_signal_with_noise
        else:
            print("CRC check failed. Data might be corrupted. Outputting corrupted signal.")
            corrupted_signal = np.random.normal(0, 0.05, len(audio_signal))  # 模擬失敗音訊
            return None, corrupted_signal, time, fsk_signal, fsk_signal_with_noise
    except Exception as e:
        print(f"Error during CRC check or data restoration: {e}")
        corrupted_signal = np.random.normal(0, 0.05, len(audio_signal))  # 模擬失敗音訊
        return None, corrupted_signal, time, fsk_signal, fsk_signal_with_noise


# 進行模擬：如果檔案存在，讀取檔案，否則使用麥克風錄音
input_file = "input.wav"
Fs = 8000

if os.path.exists(input_file):
    print(f"Reading audio from {input_file}...")
    audio_signal, Fs = sf.read(input_file)
    audio_signal = audio_signal / np.max(np.abs(audio_signal))  # 標
    # 標準化音訊
else:
    duration = 5  # 錄音持續時間
    print("Recording started. Please speak...")
    audio_signal = sd.rec(int(duration * Fs), samplerate=Fs, channels=1, dtype='float32')
    sd.wait()
    audio_signal = audio_signal.flatten()
    audio_signal = audio_signal / np.max(np.abs(audio_signal))  # 標準化

# 模擬傳輸
print("Simulating transmission...")
restored_audio_signal_with_noise, restored_audio_signal_no_noise, time, fsk_signal_no_noise, fsk_signal_with_noise = \
    simulate_fsk_transmission(audio_signal, Fs, noise=True, noise_level=0.25)

# 檢查還原後的音訊信號並播放
if restored_audio_signal_with_noise is not None and restored_audio_signal_no_noise is not None:
    print("Playing the restored audio without noise...")
    sd.play(restored_audio_signal_no_noise, Fs)
    sd.wait()

    print("Playing the restored audio with noise...")
    sd.play(restored_audio_signal_with_noise, Fs)
    sd.wait()

    # 繪製每個步驟的圖像
    # 圖 1: 原始語音信號
    plt.figure(figsize=(10, 4))
    plt.plot(audio_signal[:1000], label="Original Audio Signal")
    plt.title("Original Audio Signal")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 圖 2: A 手機數位訊號處理
    plt.figure(figsize=(10, 4))
    plt.plot(np.unpackbits(quantize_audio(audio_signal[:1000]).view(np.uint8)), label="A Phone Digital Signal")
    plt.title("A Phone Digital Signal")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 圖 3: FSK 調變信號
    plt.figure(figsize=(10, 4))
    plt.plot(time[:1000], fsk_signal_no_noise[:1000], label="FSK Modulated Signal")
    plt.title("FSK Modulated Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 圖 4: 基地台接收的信號 (無干擾)
    plt.figure(figsize=(10, 4))
    plt.plot(time[:1000], fsk_signal_no_noise[:1000], label="Base Station Received Signal (No Noise)")
    plt.title("Base Station Received Signal (No Noise)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 圖 5: 基地台發射的信號 (有干擾與無干擾)
    plt.figure(figsize=(10, 4))
    plt.plot(time[:1000], fsk_signal_no_noise[:1000], label="Base Station Transmitted Signal (No Noise)")
    plt.plot(time[:1000], fsk_signal_with_noise[:1000], label="Base Station Transmitted Signal (With Noise)",
             color='orange')
    plt.title("Base Station Transmitted Signal (With and Without Noise)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 圖 6: B 手機解調後的信號
    plt.figure(figsize=(10, 4))
    demodulated_bits = fsk_demodulate(fsk_signal_with_noise)
    plt.plot(demodulated_bits[:1000], label="B Phone Demodulated Signal")
    plt.title("B Phone Demodulated Signal")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 圖 7: B 手機還原的語音信號（無干擾）
    plt.figure(figsize=(10, 4))
    plt.plot(restored_audio_signal_no_noise[:1000], label="B Phone Restored Audio Signal (No Noise)")
    plt.title("B Phone Restored Audio Signal (No Noise)")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 圖 8: B 手機還原的語音信號（有干擾）
    plt.figure(figsize=(10, 4))
    plt.plot(restored_audio_signal_with_noise[:1000], label="B Phone Restored Audio Signal (With Noise)",
             color='orange')
    plt.title("B Phone Restored Audio Signal (With Noise)")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

else:
    print("Failed to restore audio due to CRC check failure or data corruption.")
