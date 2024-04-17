from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

# List of words corresponding to the rows
# words = ['away', 'black', 'bring', 'cabinet', 'camera', 'clockwise', 'close', 'closed', 'closer', 'container', 'counter', 'cup', 'cupboard', 'dark', 'down', 'drawer', 'dresser', 'faucet', 'from', 'front', 'glass', 'handle', 'left', 'light', 'lighter', 'move', 'mug', 'no', 'open', 'pull', 'push', 'rep', 'right', 'rotate', 'shift', 'shut', 'slide', 'spin', 't', 'tap', 'towards', 'translate', 'turn', 'un', 'valve', 'white']
words = ['button', 'close', 'door', 'drawer', 'from', 'goal', 'insert', 'joint', 'open', 'peg', 'pick', 'place', 'point', 'press', 'puck', 'push', 'reach', 'revolving', 'sideways', 'top', 'window', 'with']

data = np.load("/home/zxliang/new-code/LISA-hot/lisa/matrix_MW.npy")

# Adjusting the word cloud generation to handle skills with no non-zero word frequencies
# fig, axes = plt.subplots(5, 4, figsize=(20, 25))
# axes = axes.flatten()  # Flattening the 2D array of axes for easy indexing

for i in range(data.shape[1]):
    word_freq = {words[j]: data[j, i] for j in range(data.shape[0]) if data[j, i] > 0}

    # Only generate a word cloud if there are non-zero frequencies
    if word_freq:
        print(i+1)
        wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(word_freq)
        plt.imshow(wordcloud, interpolation="bilinear")

        plt.figure(figsize=(20, 20))
        plt.imshow(wordcloud, interpolation='bilinear')
        # plt.title(f'Skill {i + 1}', fontsize=22)
        plt.axis('off')

        # Save the word cloud image with the title
        plt.savefig(f'wordcloud2/skill_{i + 1}.png')

        # If you also want to display the image in the notebook
        # plt.show()

# plt.show()
