---
layout: post
title: Basics of Aligning Test
author: Zachary Nowak
categories: Personal
---
#### Update: 3/10/2022
I have started processing the wav files for season 1 of friends. Was able to get roughly half processed. 
The rest had words which needed preprocessing to get their pronounciation. On average it takes 5-6 minutes
to align a 12 minute wav file.

***

### how to get the pronunciation of a word?
I am created a local solution using the [CMUdict](https://github.com/cmusphinx/cmudict).

```
> conda install swig
> pip install pocketsphinx
```

Or you can directly download the .dict file.
I was able to get pronounciations of a word as follows:

```python
  import os
  from pocketsphinx import get_model_path
  
  temp = {}
  with open(os.path.join(get_model_path(), 'cmudict-en-us.dict')) as f:
    for line in f:
        key, *value = line.split()
        temp[key] = " ".join(value)

  def get_pronunciation(word):
    try:
        return temp[word]
    except KeyError:
        return ""
```

You should expect an output like this (note it is expecting lowercase and can only handle one word):
<code> get_pronunciation("hello") == "HH AH L OW"</code>

### how to handle compound and multi-word words?
This is excellently answered [here](https://stackoverflow.com/questions/33666557/get-phonemes-from-any-word-in-python-nltk-or-other-modules): 
the general solution is keep breaking down the main word into it's word parts.

```python
import nltk
from functools import lru_cache
from itertools import product as iterprod

try:
    arpabet = nltk.corpus.cmudict.dict()
except LookupError:
    nltk.download('cmudict')
    arpabet = nltk.corpus.cmudict.dict()

@lru_cache()
def wordbreak(s):
    s = s.lower()
    if s in arpabet:
        return arpabet[s]
    middle = len(s)/2
    partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in arpabet and wordbreak(suf) is not None:
            return [x+y for x,y in iterprod(arpabet[pre], wordbreak(suf))]
    return None
```

You should expect an output like this:

<code> wordbreak("heartbreaker") == [['HH', 'AA1', 'R', 'T', 'B', 'R', 'EY1', 'K', 'ER0']]</code>
### how to handle numbers?
I found the easiest solution was with a python module called [inflect](https://pypi.org/project/inflect/)
You pass the number in and it outputs it in string form.

```python
>>> import inflect
>>> p = inflect.engine()
>>> p.number_to_words(1000, andword='')
'one thousand'
>>> p.number_to_words(1045, andword='')
'one thousand forty-five'
>>> p.number_to_words(10.45, andword='') 
'ten point four five'
>>> p.number_to_words(0.5, andword='') 
'zero point five'
```

### how to handle accents?
One solution is to convert it to a standard form using [unidecode](https://pypi.org/project/Unidecode/):

```python
>>> from unidecode import unidecode
>>> unidecode(DÉJÀ)
DEJA
```

### how to align subtitles to a audio file?
Inspired by [HuthLab](https://github.com/HuthLab/p2fa-vislab)'s use of Penn Phonetics Laboratory [p2fa-vislab](https://web.sas.upenn.edu/phonetics-lab/),
I also used it. I had to update the code to *Python 3.X*. It was also a challenge to download [HTK](https://htk.eng.cam.ac.uk/) which does most of the processing.

Some Helpful Tips For Downloading HTK
- when you **make** if you get the error <code>/usr/include/stdio.h:27:10: fatal error: bits/libc-header-start.h: No such file or directory
 #include <bits/libc-header-start.h></code>
  
  Do this <code>sudo apt-get install gcc-multilib</code> or <code>sudo apt-get install g++-multilib</code> to install the missing 32 bit libraries per [this](https://stackoverflow.com/questions/54082459/fatal-error-bits-libc-header-start-h-no-such-file-or-directory-while-compili)
- when you **make** if you get the error <code>"/usr/bin/ld: cannot find -lX11" error when installing htk</code> 
  
  Do this <code>sudo apt-get install libx11-dev</code> per [this](https://stackoverflow.com/questions/40451054/cant-install-htk-on-linux)
- when you **make** if you get the error <code>"gnu/stubs-32.h: No such file or directory"</code>
  
  Do one of these per [this](https://stackoverflow.com/questions/7412548/error-gnu-stubs-32-h-no-such-file-or-directory-while-compiling-nachos-source) (There is other systems in that answer).
  - **UBUNTU** <code>sudo apt-get install libc6-dev-i386</code>
  - **CentOS 5.8** The package is <code>glibc-devel.i386</code>
  - **CentOS 6 / 7** The package is <code>glibc-devel.i686</code>
  
By this step hopefully you have HTK working!
1. Convert you subtitle file (.srt or .en.sub) to the json schema required.
  
  **.srt**
  ```text
  1
  00:00:02,877 --> 00:00:04,294
  What you guys don't understand is...
  
  2
  00:00:04,504 --> 00:00:07,548
  ...for us, kissing is as important
  as any part of it.
  ```
  
  **.en.sub**
  ```text
  {63}{127}This is pretty much|what's happened so far.
  {133}{200}{y:i}Ross was in love|{y:i}with Rachel since forever.
  ```
  
  **.json**
  ```json
  [
	  {
		  "speaker": "Narrator", 
		  "line": "What you guys don't understand is...  "
	  },
	  {
		  "speaker": "Narrator", 
		  "line": "...for us, kissing is as important as any part of it. "
	  }
  ]
  ```
  **VERY IMPORTANT** if you have it as <code>},\n]</code> instead of <code>}\n]</code> at the end it will not work!
  

2. Convert your **.mp3** (or any other type) of audio to a 16 bit and mono **.wav** file. This can be done using [ffmpeg](https://ffmpeg.org/)
  I recommend installing it without sudo access with <code>conda install -c conda-forge ffmpeg</code>
  
  ```python
  import subprocess
  import os
  import sys

  def convert_video_to_audio_ffmpeg(video_file, output_ext="wav"):
    """Converts video to audio directly using `ffmpeg` command
    with the help of subprocess module"""
    filename, ext = os.path.splitext(video_file)
    # 16 bit mono sampled 16000
    subprocess.call(["ffmpeg", "-y", "-i", video_file,"-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", f"{filename}.{output_ext}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
  ```

3. Then execute this command in the p2fa-vislab folder
  
  <code>python align.py ../input.wav ../subtitles.json ../aligned_output.json</code>

### what is the output?
```json
  {
    "words": [
        {
            "alignedWord": "WHAT",
            "start": 0.0125,
            "end": 0.1025,
            "word": "What",
            "line_idx": 0,
            "speaker": "Narrator"
        },
        {
            "alignedWord": "YOU",
            "start": 0.1025,
            "end": 1.9825,
            "word": "you",
            "line_idx": 0,
            "speaker": "Narrator"
        },
        {
            "alignedWord": "GUYS",
            "start": 1.9825,
            "end": 2.3125,
            "word": "guys",
            "line_idx": 0,
            "speaker": "Narrator"
        },
        {
            "alignedWord": "DON'T",
            "start": 2.3125,
            "end": 2.6025,
            "word": "don't",
            "line_idx": 0,
            "speaker": "Narrator"
        },
        {
            "alignedWord": "UNDERSTAND",
            "start": 2.6025,
            "end": 3.2425,
            "word": "understand",
            "line_idx": 0,
            "speaker": "Narrator"
        },
        {
            "alignedWord": "IS",
            "start": 3.2425,
            "end": 3.4825,
            "word": "is...",
            "line_idx": 0,
            "speaker": "Narrator"
         }
    ]
}
```

