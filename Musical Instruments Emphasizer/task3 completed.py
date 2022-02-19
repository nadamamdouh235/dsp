import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack as fftpk
import numpy as np
from matplotlib import pyplot as plt
import numpy.fft as fft
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
import wave
import sys
from scipy.io import wavfile
import wave
import struct
import numpy
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
from tkinter import *
import tkinter
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy import signal
from scipy.io.wavfile import read, write
import sounddevice as sd
from functools import partial
import threading
from threading import Thread
from scipy.fftpack import fft
from subprocess import call
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
from matplotlib.widgets import Slider
from scipy.signal.bsplines import cubic
import multiprocessing
import time
import scipy.fftpack as fftpk
import numpy.fft as fft
from scipy import signal
from functools import partial
from PIL import ImageTk, Image

root = tk.Tk()
style = ttk.Style(root)

# tell tcl where to find the awthemes packages
root.tk.eval("""
set base_theme_dir awthemes-10.4.0

package ifneeded awthemes 10.4.0 \
    [list source [file join $base_theme_dir awthemes.tcl]]
package ifneeded awdark 7.12 \
    [list source [file join $base_theme_dir awdark.tcl]]
""")

root.tk.call("package", "require", 'awdark')
root.tk.call("package", "require", 'awlight')
style.theme_use('awdark')

root.title("Tab Widget")
tabControl = ttk.Notebook(root)
Grid.rowconfigure(root, 0, weight=1)
Grid.columnconfigure(root, 0, weight=1)
Grid.columnconfigure(root, 1, weight=1)

equalizer_tab = ttk.Frame(tabControl)
equalizer_tab.grid(row=0, column=0, sticky="nsew")

instruments_tab = ttk.Frame(tabControl)
instruments_tab.grid(row=0, column=0, sticky="nsew")

tabControl.add(equalizer_tab, text='equalizer')
tabControl.add(instruments_tab, text='instruments')

Grid.rowconfigure(equalizer_tab, 0, weight=1)
Grid.columnconfigure(equalizer_tab, 0, weight=1)
Grid.rowconfigure(instruments_tab, 0, weight=1)
Grid.columnconfigure(instruments_tab, 0, weight=1)
tabControl.grid(row=0, column=0, columnspan=2, sticky="nsew")


#-----------------------------------Equalizer tab------------------------------------#

audio_plot = plt.figure(figsize=(5, 5))
spectrogram_plot = plt.figure(figsize=(5, 5))

audio_axis = audio_plot.add_axes([0.1, 0.1, 0.8, 0.8])
spectrogram_axis = spectrogram_plot.add_axes([0.1, 0.1, 0.8, 0.8])

Grid.rowconfigure(equalizer_tab, 0, weight=1)
Grid.columnconfigure(equalizer_tab, 0, weight=1)
Grid.columnconfigure(equalizer_tab, 1, weight=1)
Grid.rowconfigure(equalizer_tab, 1, weight=1)
Grid.rowconfigure(equalizer_tab, 2, weight=1)
Grid.rowconfigure(equalizer_tab, 3, weight=1)


browse_frame=LabelFrame(equalizer_tab, width=300,height=300, background='#33393B')
browse_frame.grid(row=0, column=0, sticky="nw")

graph_frame = LabelFrame(equalizer_tab, width=300,height=300, background='#33393B')
graph_frame.grid(row=1, column=0, sticky="nsew")

spectrogram_frame = LabelFrame(equalizer_tab, width=300, height=300, background="#33393B")
spectrogram_frame.grid(row=1, column=1,rowspan=2, sticky="nsew")

audio_control_frame=LabelFrame(equalizer_tab, width=300,height=300, background='#33393B')
audio_control_frame.grid(row=2, column=0, sticky="nsew")

equalizer_control_frame = LabelFrame(equalizer_tab, width=300,height=300, background='#33393B')
equalizer_control_frame.grid(row=3, column=0, sticky="nsew",columnspan=2)






root.configure(bg=style.lookup('TFrame', 'background'))

Grid.rowconfigure(graph_frame, 0, weight=1)
Grid.columnconfigure(graph_frame, 0, weight=1)

Grid.rowconfigure(spectrogram_frame, 0, weight=1)
Grid.columnconfigure(spectrogram_frame, 0, weight=1)

Grid.rowconfigure(audio_control_frame, 0, weight=1)
Grid.columnconfigure(audio_control_frame, 0, weight=1)

# A tk.DrawingArea.
audio_canvas = FigureCanvasTkAgg(audio_plot, master=graph_frame)
audio_canvas.draw()
audio_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
# A tk.DrawingArea.
spectrogram_canvas = FigureCanvasTkAgg(
    spectrogram_plot, master=spectrogram_frame)
spectrogram_canvas.draw()
spectrogram_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

is_inverse = False
is_stopped = False
counter = 0
Time = 0
original_data = 0
modified_data = 0
sampleRate = 0


def animation(i):
    global counter
    global Time
    if not is_stopped:
        if (counter*1000+sampleRate < len(original_data)):
            audio_axis.set_xlim(Time[1000*counter],
                                Time[1000*counter+sampleRate])
            counter = counter+1


def animate(original_data):
    global counter
    global anim
    counter = 0
    audio_axis.clear()
    audio_axis.plot(Time, original_data)
    anim = FuncAnimation(audio_plot, func=animation, interval=10)
    anim.save('mygif.gif')


def spectrogram(signal, rate):
    spectrogram_axis.cla()
    plt.title('Spectrogram')
    plt.specgram(signal, Fs=rate, NFFT=512)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.ylim(0, 5000)
    spectrogram_canvas.draw()


def sing(signal, rate):
    sd.play(signal, rate)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
current_VolumeDb = volume.GetMasterVolumeLevel()


def change_volume(event):
    volume_value = volume_slider.get()
    volume.SetMasterVolumeLevel(0.6525*volume_value-65.25, None)


# browse for audio file, play it and plot its signal and spectrogram
def browse():
    global Time
    global original_data
    global sampleRate
    global music_thread
    file_path = filedialog.askopenfilename()
    sampleRate, original_data = wavfile.read(file_path)
    duration_seconds = len(original_data) / float(sampleRate)
    Time = np.linspace(1, duration_seconds, len(original_data))
    spectrogram(original_data, sampleRate)
    animate(original_data)
    # thread to play the audio
    music_thread = Thread(target=sing(original_data, sampleRate))
    music_thread.start()


# pause plotting signal gragh
def pause():
    global anim
    global is_stopped
    is_stopped = True


# resume playing signal graph
def play():
    global is_stopped
    global anim
    is_stopped = False


# converting the audio signal to fourier domain to change its properties
def fourier_transform(self):
    global anim
    is_inverse = True
    sd.stop()
    anim.event_source.stop()
    data_after_fourier = fft.rfft(original_data)
    data_after_fourier = data_after_fourier / 2.0**15
    data_after_fourier.shape
    data_after_fourier = data_after_fourier.tolist()

    # get frequencies of the audio signal
    frequancy = fft.rfftfreq(len(original_data), (1.0/sampleRate))
    frequancy = frequancy.tolist()

    # specifying frequency ranges of instruments in the audio signal
    # [guitar, piano, drums, violin, organ]
    minFreq = [1150, 460, 0, 5050, 6789]
    maxFreq = [5000, 900, 400, 5800, 7931]
    gain = []
    gain.append(guitar_slider.get())
    gain.append(piano_slider.get())
    gain.append(drums_slider.get())
    gain.append(violin_slider.get())
    gain.append(organ_slider.get())

    # amplify each frequency range with its gain
    for i in range(len(minFreq)):
        for j in range(len(frequancy)):
            if (frequancy[j] >= minFreq[i] and frequancy[j] <= maxFreq[i]):
                data_after_fourier[j] = data_after_fourier[j] * gain[i]

    inverse_fourier(data_after_fourier)


# convert signal from frequency domain back to time domain, play its audio and plot its signal and spectrogram
def inverse_fourier(data_after_fourier):
    global sampleRate
    global modified_data
    global Time
    global music_thread
    inverse_data = np.fft.irfft(data_after_fourier)
    write("test.wav", sampleRate, inverse_data)
    sampleRate, modified_data = wavfile.read("test.wav")
    new_duration = len(modified_data) / float(sampleRate)
    Time = np.linspace(1, new_duration, len(modified_data))
    spectrogram(modified_data, sampleRate)
    animate(modified_data)
    music_thread = Thread(target=sing(modified_data, sampleRate))
    music_thread.start()


# pause audio
def stop_song():
    global modified_data
    global song_after_pause
    global counter
    global is_inverse
    sd.stop()
    signal = original_data
    if (is_inverse == True):
        signal = modified_data

    # get position where audio was paused
    current_count = counter*1000+sampleRate
    song_after_pause = signal[current_count:]


# play audio starting from the paused position
def play_after_pause():
    global song_after_pause
    sd.play(song_after_pause, sampleRate)


#--------------------------------Sliders and buttons--------------------------------#

Grid.rowconfigure(equalizer_control_frame, 2, weight=1)
Grid.rowconfigure(equalizer_control_frame, 3, weight=1)
Grid.columnconfigure(equalizer_control_frame, 0, weight=1)
Grid.columnconfigure(equalizer_control_frame, 1, weight=1)
Grid.columnconfigure(equalizer_control_frame, 2, weight=1)
Grid.columnconfigure(equalizer_control_frame, 3, weight=1)
Grid.columnconfigure(equalizer_control_frame, 4, weight=1)
Grid.rowconfigure(audio_control_frame, 0, weight=1)
Grid.columnconfigure(audio_control_frame, 0, weight=1)
Grid.columnconfigure(audio_control_frame, 1, weight=1)
Grid.columnconfigure(audio_control_frame, 2, weight=1)

guitar_photo = PhotoImage(file=r"guitaricon.png")
guitar_label_photo = Label(equalizer_control_frame, text="guitar",image=guitar_photo, bd=5, justify=RIGHT, padx=45, pady=440)
guitar_label_photo.grid(row=2, column=0, sticky="nsew")

guitar_slider = ttk.Scale(equalizer_control_frame, from_=10, to=0,orient=VERTICAL, length=100)
guitar_slider.set(1)
guitar_slider.bind("<ButtonRelease-1>", fourier_transform)
guitar_slider.grid(row=3, column=0, sticky="ns")

piano_photo = PhotoImage(file=r"piano.png")
piano_label_photo = Label(equalizer_control_frame, text="piano",
                          image=piano_photo, bd=5, justify=RIGHT, padx=45, pady=440)
piano_label_photo.grid(row=2, column=1, sticky="nsew")

piano_slider = ttk.Scale(equalizer_control_frame, from_=10, to=0,
                         orient=VERTICAL, length=100)
piano_slider.set(1)
piano_slider.bind("<ButtonRelease-1>", fourier_transform)
piano_slider.grid(row=3, column=1, sticky="ns")

drums_photo = PhotoImage(file=r"drums.png")
drums_label_photo = Label(equalizer_control_frame, text="piano",
                          image=drums_photo, bd=5, justify=RIGHT, padx=45, pady=440)
drums_label_photo.grid(row=2, column=2, sticky="nsew")

drums_slider = ttk.Scale(equalizer_control_frame, from_=10, to=0,
                         orient=VERTICAL, length=100)
drums_slider.set(1)
drums_slider.bind("<ButtonRelease-1>", fourier_transform)
drums_slider.grid(row=3, column=2, sticky="ns")

violin_photo = PhotoImage(file=r"violin.png")
violin_label_photo = Label(equalizer_control_frame, text="piano",
                           image=violin_photo, bd=5, justify=RIGHT, padx=45, pady=440)
violin_label_photo.grid(row=2, column=3, sticky="nsew")

violin_slider = ttk.Scale(equalizer_control_frame, from_=10, to=0,
                          orient=VERTICAL, length=100)
violin_slider.set(1)
violin_slider.bind("<ButtonRelease-1>", fourier_transform)
violin_slider.grid(row=3, column=3, sticky="ns")

organ_photo = PhotoImage(file=r"organ.png")
organ_label_photo = Label(equalizer_control_frame, text="piano",
                          image=organ_photo, bd=5, justify=RIGHT, padx=45, pady=440)
organ_label_photo.grid(row=2, column=4, sticky="nsew")

organ_slider = ttk.Scale(equalizer_control_frame, from_=10, to=0,
                         orient=VERTICAL, length=100)
organ_slider.set(1)
organ_slider.bind("<ButtonRelease-1>", fourier_transform)
organ_slider.grid(row=3, column=4, sticky="ns")

button_browse = ttk.Button(master=browse_frame, text="browse for audio", command=browse)
button_browse.grid(row=0, column=1, sticky="nw")



start_photo = PhotoImage(file=r'start1.png')
pause_photo = PhotoImage(file=r'pause3.png')
button_play = ttk.Button(master=audio_control_frame, text="play", command=lambda: [
                         play(), play_after_pause()], image=start_photo)
button_play.grid(row=0, column=0, sticky="new")

buttonpause = ttk.Button(master=audio_control_frame, text="pause", command=lambda: [
                         pause(), stop_song()], image=pause_photo)
buttonpause.grid(row=0, column=1, sticky="new")

volume_slider = ttk.Scale(audio_control_frame, orient='horizontal',
                          from_=0, to=100, command=change_volume,length=200)
volume_slider.set(50)
volume_slider.grid(padx=(180,0),pady=20,row=0, column=2, sticky="ne")

#------------------------instruments_tab---------------------------------#
Grid.rowconfigure(instruments_tab, 0, weight=1)
Grid.rowconfigure(instruments_tab, 1, weight=1)
Grid.columnconfigure(instruments_tab, 0, weight=1)

buttons_frame = LabelFrame(instruments_tab, text='',width=300,height=200,
                           font=20, background='#33393B')
buttons_frame.grid(row=0, column=0,sticky="we")

piano_frame = LabelFrame(instruments_tab, text='PIANO',width=300,height=300,
                         fg='white', font=20, background='#33393B')
piano_frame.grid(row=1, column=0,sticky='nsew')

xylophone_frame = LabelFrame(instruments_tab, text='XYLOPHONE',width=300,height=300,
                             fg='white', font=20, background='#33393B')
xylophone_frame.grid(row=1, column=0)

drums_frame = LabelFrame(instruments_tab, text='DRUMS',width=300,height=300,
                         fg='white', font=20, background='#33393B')
drums_frame.grid(row=1, column=0)

def on_click(controlled_frame,uncontrolled_frame1,uncontrolled_frame2,toggle_flag):
    
    if toggle_flag:
        controlled_frame.grid_forget()
        toggle_flag = False
    else:
        controlled_frame.grid(row=1, column=0, sticky="nsew")
        uncontrolled_frame1.grid_forget()
        uncontrolled_frame2.grid_forget()
        toggle_flag = True
    return toggle_flag



instruments_sample_rate = 44100

#--------------------------------------PIANO-------------------------------------#
octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']


def get_wave(piano_freq, duration=0.5):
    piano_amplitude = 4096
    piano_time = np.linspace(0, duration, int(
        instruments_sample_rate * duration))
    piano_wave = piano_amplitude*np.sin(2*np.pi*piano_freq*piano_time)
    return piano_wave


def get_piano_notes():
    base_frequency = 261.63
    note_frequencies = {octave[i]: base_frequency *
                        pow(2, (i/12)) for i in range(len(octave))}
    note_frequencies[''] = 0.0
    return note_frequencies


def get_song_data(music_notes):
    note_frequencies = get_piano_notes()
    note = [get_wave(note_frequencies[note])
            for note in music_notes.split('-')]
    note = np.concatenate(note)

    return note.astype(np.int16)


def music_notes(note):
    music_notes = octave[note]
    note_data = get_song_data(music_notes)
    sd.play(note_data)

#------------------------------piano buttons--------------------------------#
Grid.rowconfigure(piano_frame, 2, weight=1)
Grid.rowconfigure(piano_frame, 3, weight=1)
for i in range (0,7):
    Grid.columnconfigure(piano_frame, i, weight=2)



show_piano = False
piano_frame.grid_forget()

Grid.rowconfigure(buttons_frame, 0, weight=1)
Grid.columnconfigure(buttons_frame, 0, weight=1)

piano_button_image = PhotoImage(file=r'piano button.png')
piano_button = ttk.Button(buttons_frame, text="Piano",image=piano_button_image,
command=lambda:show_piano==on_click(piano_frame,drums_frame,xylophone_frame,show_piano)).grid(row=0, column=0, sticky="nsew")
#-----------------------------------xylophone-----------------------------#


xylophone_freq = [700, 800, 900, 1000, 1100,
                  1200, 1300, 1400, 1500, 1600, 1700, 1800]


def xylophone(xylophone_music_notes):
    global xylophone_wave
    xylophone_w = 2 * np.pi * xylophone_music_notes
    xylophone_time = np.linspace(0, 0.9, int(instruments_sample_rate*0.9))
    xylophone_amplitude = np.exp(-0.0020*xylophone_w*xylophone_time)
    xylophone_wave = np.sin(xylophone_w*xylophone_time)*xylophone_amplitude


def xylophone_notes(event, xylophone_note):
    xylophone_music_notes = xylophone_freq[xylophone_note]
    xylophone(xylophone_music_notes)
    sd.play(xylophone_wave, instruments_sample_rate, blocking=True)

#----------------------------------xylophone image and buttons---------------------#

Grid.rowconfigure(xylophone_frame, 0, weight=1)
Grid.columnconfigure(xylophone_frame, 0, weight=1)

xylophone_bf_resize = ImageTk.PhotoImage(file="xylophone9.png")
xylophone_canvas = Canvas(xylophone_frame, width=500, height=500)
xylophone_canvas.grid(row=0,column=0, sticky='nsew')
xylophone_canvas.create_image(0, 0, image=xylophone_bf_resize, anchor='nw')

global show_xylophone
show_xylophone = False
xylophone_frame.grid_forget()


Grid.columnconfigure(buttons_frame, 1, weight=1)
xylophone_button_image = PhotoImage(file=r'xylophone button.png')

xylphone_button = ttk.Button(buttons_frame, text="xylophone",image=xylophone_button_image,
                             command=lambda:show_xylophone==on_click(xylophone_frame,drums_frame,piano_frame,show_xylophone)).grid(row=0, column=1, sticky="nsew")

#-----------------------------------DRUMS--------------------------------#
Grid.rowconfigure(drums_frame, 0, weight=1)
Grid.rowconfigure(drums_frame, 1, weight=1)
Grid.columnconfigure(drums_frame, 0, weight=1)

drum_frequency = 2000


def kick_drum(self):
    global kick_drum_wave
    kick_drum_time = np.linspace(0, 1, int(instruments_sample_rate*0.6))
    exponential = np.exp(-70*kick_drum_time)
    kick_drum_w = drum_frequency*exponential*kick_drums_slider.get()
    kick_drum_wave = np.sin(kick_drum_w*kick_drum_time)


def play_sound(event):
    sd.play(kick_drum_wave, instruments_sample_rate, blocking=True)


drums_bf_resize = ImageTk.PhotoImage(file="drum3.png")
drums_canvas = Canvas(drums_frame, width=500, height=500)
drums_canvas.grid(row=0,column=0,sticky='nsew')
drums_canvas.create_image(0, 0, image=drums_bf_resize, anchor='nw')

kick_drums_slider = ttk.Scale(drums_frame, from_=1, to=5,
                              orient=HORIZONTAL, length=500,  command=kick_drum)
kick_drums_slider.set(1)
kick_drums_slider.grid(row=1, column=0, sticky="ew")

root.bind('k', lambda event: play_sound(event))

show_drum = False
drums_frame.grid_forget()



buttons=['C','D','E','F','G','A','B','d','f','g','a','k']
j=7
for i in range (0,11):
    if (i<7):
        row_position=3
        button_width=4
        padx=40
        j=j-1
        sticky="nsew"
        button_color='white'
        font_color='black'
    else :
        row_position=2
        button_width=2
        padx= 30
        j=j+1
        sticky="ns"
        button_color='black'
        font_color='white'

    upper_piano_button=Button(piano_frame, bg=button_color,fg=font_color, text=buttons[j], command=partial(
    music_notes, i), height=10, width=button_width, borderwidth=5)
    upper_piano_button.grid(row=row_position, column=j,ipadx=padx, sticky=sticky) 

root.bind('C', lambda event: xylophone_notes(event, xylophone_note=0))

for i in range(0,11):
    root.bind(buttons[i], lambda event: xylophone_notes(event, xylophone_note=i))

Grid.columnconfigure(buttons_frame, 3, weight=1)
drums_button_image = PhotoImage(file=r'drums button.png')
drums_button = ttk.Button(buttons_frame, text="drums",image=drums_button_image,
                          command=lambda:show_drum==on_click(drums_frame,xylophone_frame,piano_frame,show_drum)).grid(row=0, column=3, sticky="nsew")
def resize(self):
    global xylophone_tobe_resized, resize_xylophone, resized_xylophone
    # open image to resize it
    xylophone_tobe_resized = Image.open("xylophone8.png")
    resize_xylophone = xylophone_tobe_resized.resize((xylophone_frame.winfo_width(),xylophone_frame.winfo_height()),
                         Image.ANTIALIAS)
    
    resized_xylophone = ImageTk.PhotoImage(resize_xylophone)
    xylophone_canvas.create_image(0, 0, image=resized_xylophone, anchor='nw')

    global drums_tobe_resized, resize_drums, resized_drums
    # open image to resize it
    drums_tobe_resized = Image.open("drum3.png")
    resize_drums = drums_tobe_resized.resize((drums_frame.winfo_width(),drums_frame.winfo_height()),
                         Image.ANTIALIAS)
    
    resized_drums = ImageTk.PhotoImage(resize_drums)
    drums_canvas.create_image(0, 0, image=resized_drums, anchor='nw')

    root.update_idletasks()


root.bind("<Configure>", resize)
tkinter.mainloop()
