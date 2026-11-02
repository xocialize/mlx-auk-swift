#!/bin/zsh
# sweep_phrasing.sh — instruction phrasings over seeds 0–4 through `auk-gates render` (the pipeline, no engine), to
# separate a template effect from seed noise when the cookbook verifier fails a case (AB-T-0416). Score with
# score_sweep.py. Keys (<case>-<phrasing>):
#   1.2-pkg / 1.2-cb     instruct TTS: the package's v0.2.0 EN template / the cookbook example's (= the V0 oracle's)
#   4.5-pkg / 4.5-cb     to-whisper on the zh whisper sample: EN template / ZH template (the cookbook example)
#   4.5en-pkg / 4.5en-zh to-whisper on an EN take (emotion-edit/en-1-input): EN template / ZH template
#   fw-pkg / fw-zh       whisper-to-normal on the zh whisper sample: EN template / ZH template
#
#   sweep_phrasing.sh <weights> <assets> <out> [variants="flash base"] [keys="1.2-pkg 1.2-cb 4.5-pkg 4.5-cb"]
here=${0:A:h}
G=${AUK_GATES:-$here/../../.build/release/auk-gates}
W=$1 A=$2 OUT=$3 variants=(${=${4:-flash base}}) keys=(${=${5:-1.2-pkg 1.2-cb 4.5-pkg 4.5-cb}})
D='一位二十多岁的女性，面对刚到家的伴侣，以温柔、关心且略带撒娇的语气轻声诉说。她的声音柔和亲密，语速稍缓，音量适中，音色甜美自然，在短语结尾处语调温暖地上扬。她正向伴侣问候回家，并温柔询问今日工作如何。'
T='Welcome home, how was work today?'
typeset -A I
I[1.2-pkg]="Generate speech based on the following description: \"$D\". The content to speak is: \"$T\"."
I[1.2-cb]="Based on the following description: \"$D\", generate speech content \"$T\"."
I[4.5-pkg]="Convert this speech into a soft whisper while preserving the speaker and content."
I[4.5-cb]="用小声耳语的方式把这段话说出来。"
I[4.5en-pkg]=${I[4.5-pkg]}
I[4.5en-zh]=${I[4.5-cb]}
I[fw-pkg]="Convert this whispered speech into a normal speaking voice while preserving the speaker and content."
I[fw-zh]="把这段耳语转换成正常说话的声音。"
for v in $variants; do
  model=$W/$([[ $v == base ]] && echo AuK || echo AuK-Flash)-bf16
  out=$OUT/$v; mkdir -p $out
  for k in $keys; do
    for s in 0 1 2 3 4; do
      case $k in
        1.2-*) inp=-; secs=1.7 ;;
        4.5en-*) inp=$A/emotion-edit/en-1-input.wav; secs=- ;;
        *) inp=$A/whisper/wh-w2n-zh-input.wav; secs=- ;;
      esac
      $G --dtype bfloat16 render $model $W/Qwen2.5-Omni-3B-Thinker-bf16 "${I[$k]}" $inp $secs $out/$k-s$s.wav $s 2>&1 | grep -E "wrote|ERROR"
    done
  done
done
