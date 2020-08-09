# Identify-Speakers-in-Cocktail-Parties-with-E2E-Attention
This project uses attention methods to build a source separation and speaker identification pipeline. It outperforms state-of-the-art methods in single-channel mixture speaker identification by far.<br>
Dataset is Hub-4.

Paper accepted interspeech 2020 for presentation. Arxiv link: https://arxiv.org/abs/2005.11408

\begin{table}
  \caption{Results; column $M/N$ lists \% of test data where at least $M$ talkers out of $N$-talker mixture are correctly identified.}
  \label{tab:results}
  \centering
  \begin{tabular}{|p{1.2cm}|c|c|c|c|c|c|}\hline
     \multicolumn{4}{|c|}{(\# spkr corr)/(\# in mixture)}&(s)&($10^9$)&($10^6$)\\
    Algorithm & 1/2 & 2/2 & 3/3&time&FLOPS&\#params \\\hline\hline
    Proposed, LSTM&99.9& 93.6 &77.7&41&41.1&12.4\\\hline
    Proposed, ResNet &99.9& 93.9 &81.2&47&158.6&20.1\\\hline
    TasNet + SincNet & 99.7 & 91.0 &74.1&52&23.8&23.0\\\hline
    Wang2018 & 95.2 & 36.3 & 12.5 & 2 & 0.15 & 1.36\\\hline
    Ablation & 99.8 & 92.1 &-&50&162.1 &17.4\\\hline
  \end{tabular}
\end{table}
