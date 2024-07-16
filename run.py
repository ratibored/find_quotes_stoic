import pandas as pd
import numpy as np
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import PySimpleGUI as sg
import os

# Загружаем модель и данные
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
file_path = os.path.join(os.path.dirname(__file__), 'data', 'embedded_citations.pkl')
df = pd.read_pickle(file_path)

# Функция для получения похожих цитат
def get_similar_quotes(request):
    user_citation = model.encode(request)
    df_ = df.copy()
    ans = []
    for stoic_citation in df_.embedded:
        result = round(util.pytorch_cos_sim(stoic_citation, user_citation).item()*100, 1)
        ans.append(result)
    df_['similarity'] = ans
    df_ = df_.sort_values('similarity', ascending=False).head(10).drop('embedded', axis=1)
    
    for i in range(10):
        text_elem_a = window[f'-author_{i+1}-']
        text_elem_q = window[f'-quote_{i+1}-']
        text_elem_s = window[f'-similarity_{i+1}-']
        
        text_elem_a.update(f"Автор: {df_.iloc[i].author}")
        text_elem_q.update(f"Цитата: {df_.iloc[i].quote}")
        text_elem_s.update(f"Схожесть: {df_.iloc[i].similarity}%")
    
    return df_.to_dict(orient='records')
    
    
    
sg.theme('DarkTeal9')

# Создаем layout с 10 блоками для вывода цитат
input_layout = [
    [sg.Text('Введите цитату: '), sg.InputText(key='-INPUT-')],
    [sg.Button('Отправить', enable_events=True, key='-FUNCTION-', font='Helvetica 14')],
]

quotes_layout = []

# Добавляем 10 блоков для отображения цитат
for i in range(10):
    quotes_layout.extend([
        [sg.Text(f'Цитата {i+1}:', font='Helvetica 16')],
        [sg.Text('', key=f'-author_{i+1}-', font='Helvetica 16')],
        [sg.Multiline('', key=f'-quote_{i+1}-', font='Helvetica 16', size=(100, 6), no_scrollbar=True, disabled=True, wrap_lines=True)],
        [sg.Text('', key=f'-similarity_{i+1}-', font='Helvetica 16')],
        [sg.HorizontalSeparator()]
    ])

# Создаем скролируемую колонку для цитат
quotes_column = sg.Column(quotes_layout, scrollable=True, vertical_scroll_only=True, size=(1800, 1000))

layout = input_layout + [[quotes_column]]

window = sg.Window('Поиск похожих стоических цитат', layout, resizable=True, size=(800, 600))

# Основной цикл обработки событий
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == '-FUNCTION-':
        get_similar_quotes(values['-INPUT-'])

window.close()
