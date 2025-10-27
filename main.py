import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from janome.tokenizer import Tokenizer
from collections import Counter, defaultdict
import networkx as nx
import re
from io import StringIO
import json

# Groq API (無料枠あり)
import requests

st.set_page_config(page_title="授業分析システム", layout="wide", page_icon="📚")

# セッション状態の初期化
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'segments' not in st.session_state:
    st.session_state.segments = None

class ClassroomAnalyzer:
    def __init__(self, custom_dict=None):
        self.tokenizer = Tokenizer()
        self.custom_dict = custom_dict or {}
        
    def load_custom_dictionary(self, dict_df):
        """ユーザー辞書を読み込む"""
        custom_dict = {}
        for _, row in dict_df.iterrows():
            word = str(row.iloc[0]).strip()
            reading = str(row.iloc[1]).strip() if len(row) > 1 else word
            custom_dict[word] = reading
        return custom_dict
    
    def tokenize_with_custom_dict(self, text):
        """カスタム辞書を優先して形態素解析"""
        tokens = []
        remaining_text = text
        
        # カスタム辞書の語を優先的に検出
        for word in sorted(self.custom_dict.keys(), key=len, reverse=True):
            if word in remaining_text:
                parts = remaining_text.split(word)
                new_remaining = []
                for i, part in enumerate(parts):
                    if i > 0:
                        tokens.append({
                            'surface': word,
                            'reading': self.custom_dict[word],
                            'pos': 'カスタム'
                        })
                    new_remaining.append(part)
                remaining_text = '###CUSTOM###'.join(new_remaining)
        
        # 残りのテキストを通常の形態素解析
        text_parts = remaining_text.split('###CUSTOM###')
        final_tokens = []
        token_idx = 0
        
        for part in text_parts:
            if part:
                for token in self.tokenizer.tokenize(part):
                    parts = str(token).split('\t')
                    surface = parts[0]
                    features = parts[1].split(',') if len(parts) > 1 else []
                    
                    final_tokens.append({
                        'surface': surface,
                        'reading': features[7] if len(features) > 7 else surface,
                        'pos': features[0] if features else '未知語'
                    })
            
            if token_idx < len(tokens):
                final_tokens.append(tokens[token_idx])
                token_idx += 1
        
        return final_tokens
    
    def analyze_speakers(self, df):
        """話者ごとの発言を分析"""
        speaker_analysis = defaultdict(lambda: {'utterances': [], 'word_freq': Counter()})
        
        for _, row in df.iterrows():
            speaker = str(row['Speaker']).strip()
            utterance = str(row['Utterance']).strip()
            
            tokens = self.tokenize_with_custom_dict(utterance)
            words = [t['surface'] for t in tokens if t['pos'] in ['名詞', '動詞', '形容詞', 'カスタム']]
            
            speaker_analysis[speaker]['utterances'].append(utterance)
            speaker_analysis[speaker]['word_freq'].update(words)
        
        return speaker_analysis
    
    def segment_classroom(self, df, groq_api_key):
        """AIを使って授業をセグメントに分割"""
        # 授業記録を結合
        full_text = ""
        for idx, row in df.iterrows():
            full_text += f"[{row['No']}] {row['Speaker']}: {row['Utterance']}\n"
        
        # Groq APIでセグメント分析
        prompt = f"""以下の授業記録を分析し、テーマや内容の変化に基づいて3〜7個のセグメント（意味のあるまとまり）に分けてください。
各セグメントには以下の情報を含めてください：
- segment_id: セグメント番号（1から開始）
- start_no: 開始発言番号
- end_no: 終了発言番号
- theme: セグメントのテーマ（20文字以内）
- summary: セグメントの要約（50文字以内）

JSON形式で出力してください。

授業記録:
{full_text[:3000]}

出力形式:
{{"segments": [{{"segment_id": 1, "start_no": 1, "end_no": 5, "theme": "導入", "summary": "授業の目標を説明"}}]}}
"""
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # JSONを抽出
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    segments_data = json.loads(json_match.group())
                    return segments_data.get('segments', [])
            
        except Exception as e:
            st.warning(f"AI分析エラー: {e}. デフォルトセグメント分割を使用します。")
        
        # フォールバック: 均等分割
        total_rows = len(df)
        segment_size = max(total_rows // 5, 1)
        segments = []
        for i in range(0, total_rows, segment_size):
            segments.append({
                'segment_id': len(segments) + 1,
                'start_no': int(df.iloc[i]['No']),
                'end_no': int(df.iloc[min(i + segment_size - 1, total_rows - 1)]['No']),
                'theme': f'セグメント {len(segments) + 1}',
                'summary': '自動分割'
            })
        
        return segments
    
    def analyze_segments(self, df, segments):
        """各セグメントの主要語を分析"""
        segment_analysis = []
        
        for seg in segments:
            seg_df = df[(df['No'] >= seg['start_no']) & (df['No'] <= seg['end_no'])]
            
            all_words = []
            for _, row in seg_df.iterrows():
                tokens = self.tokenize_with_custom_dict(str(row['Utterance']))
                words = [t['surface'] for t in tokens if t['pos'] in ['名詞', '動詞', '形容詞', 'カスタム'] and len(t['surface']) > 1]
                all_words.extend(words)
            
            word_freq = Counter(all_words)
            top_words = word_freq.most_common(20)
            
            segment_analysis.append({
                'segment_id': seg['segment_id'],
                'theme': seg['theme'],
                'summary': seg['summary'],
                'top_words': top_words,
                'total_words': len(all_words),
                'unique_words': len(set(all_words))
            })
        
        return segment_analysis
    
    def analyze_word_transitions(self, df, segments, segment_analysis):
        """セグメント間の語の遷移を分析"""
        transitions = []
        
        for i in range(len(segment_analysis) - 1):
            current_seg = segment_analysis[i]
            next_seg = segment_analysis[i + 1]
            
            current_words = set([w[0] for w in current_seg['top_words'][:10]])
            next_words = set([w[0] for w in next_seg['top_words'][:10]])
            
            # 共通語（引き継がれた語）
            common_words = current_words & next_words
            
            # 影響力スコア
            influence_score = len(common_words) / len(current_words) if current_words else 0
            
            transitions.append({
                'from_segment': current_seg['segment_id'],
                'to_segment': next_seg['segment_id'],
                'common_words': list(common_words),
                'influence_score': influence_score
            })
        
        return transitions

# Streamlit UI
st.title("📚 授業分析システム")
st.markdown("形態素解析とAIを活用した授業記録の分析ツール")

# サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    
    # Groq APIキー入力
    groq_api_key = st.text_input(
        "Groq APIキー（無料）",
        type="password",
        help="https://console.groq.com でAPIキーを取得できます（無料枠あり）"
    )
    
    st.markdown("---")
    st.markdown("### 📖 使い方")
    st.markdown("""
    1. Groq APIキーを入力
    2. 授業記録CSVをアップロード
    3. （オプション）カスタム辞書をアップロード
    4. 分析を実行
    """)

# メインエリア
tab1, tab2, tab3, tab4 = st.tabs(["📁 データ読み込み", "👥 話者分析", "📊 セグメント分析", "🔄 語の遷移"])

with tab1:
    st.header("データ読み込み")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("授業記録ファイル")
        classroom_file = st.file_uploader(
            "CSVファイルをアップロード（No, Speaker, Utterance）",
            type=['csv'],
            key='classroom'
        )
        
        if classroom_file:
            try:
                df = pd.read_csv(classroom_file)
                st.success(f"✅ {len(df)}行のデータを読み込みました")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"エラー: {e}")
                df = None
    
    with col2:
        st.subheader("カスタム辞書（オプション）")
        dict_file = st.file_uploader(
            "CSVファイルをアップロード（語, 読み方）",
            type=['csv'],
            key='dictionary'
        )
        
        custom_dict_df = None
        if dict_file:
            try:
                custom_dict_df = pd.read_csv(dict_file, header=None)
                st.success(f"✅ {len(custom_dict_df)}語の辞書を読み込みました")
                st.dataframe(custom_dict_df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"エラー: {e}")
    
    st.markdown("---")
    
    if st.button("🚀 分析を開始", type="primary", use_container_width=True):
        if not groq_api_key:
            st.error("Groq APIキーを入力してください")
        elif classroom_file is None:
            st.error("授業記録ファイルをアップロードしてください")
        else:
            with st.spinner("分析中..."):
                # アナライザー初期化
                analyzer = ClassroomAnalyzer()
                
                # カスタム辞書読み込み
                if custom_dict_df is not None:
                    analyzer.custom_dict = analyzer.load_custom_dictionary(custom_dict_df)
                    st.info(f"カスタム辞書: {len(analyzer.custom_dict)}語を適用")
                
                # 話者分析
                speaker_analysis = analyzer.analyze_speakers(df)
                
                # セグメント分割
                segments = analyzer.segment_classroom(df, groq_api_key)
                
                # セグメント分析
                segment_analysis = analyzer.analyze_segments(df, segments)
                
                # 語の遷移分析
                transitions = analyzer.analyze_word_transitions(df, segments, segment_analysis)
                
                # セッション状態に保存
                st.session_state.analyzed_data = {
                    'df': df,
                    'speaker_analysis': speaker_analysis,
                    'segments': segments,
                    'segment_analysis': segment_analysis,
                    'transitions': transitions,
                    'analyzer': analyzer
                }
                
                st.success("✅ 分析が完了しました！他のタブで結果を確認してください。")

with tab2:
    st.header("👥 話者分析")
    
    if st.session_state.analyzed_data:
        data = st.session_state.analyzed_data
        speaker_analysis = data['speaker_analysis']
        
        st.subheader("話者ごとの主張と特徴")
        
        for speaker, info in speaker_analysis.items():
            with st.expander(f"🗣️ {speaker} （発言数: {len(info['utterances'])}）"):
                st.markdown("**主要な語:**")
                top_words = info['word_freq'].most_common(15)
                
                # 語の頻度を棒グラフで表示
                if top_words:
                    words_df = pd.DataFrame(top_words, columns=['語', '頻度'])
                    fig = px.bar(words_df, x='語', y='頻度', title=f'{speaker}の主要語')
                    st.plotly_chart(fig, use_container_width=True, key=f"speaker_chart_{speaker}")
                
                st.markdown("**発言例:**")
                for i, utterance in enumerate(info['utterances'][:3], 1):
                    st.markdown(f"{i}. {utterance}")
    else:
        st.info("まず「データ読み込み」タブで分析を実行してください。")

with tab3:
    st.header("📊 セグメント分析")
    
    if st.session_state.analyzed_data:
        data = st.session_state.analyzed_data
        segments = data['segments']
        segment_analysis = data['segment_analysis']
        
        # セグメント関係図
        st.subheader("セグメントの流れ")
        
        G = nx.DiGraph()
        for seg in segment_analysis:
            G.add_node(seg['segment_id'], label=seg['theme'])
        
        for i in range(len(segment_analysis) - 1):
            G.add_edge(segment_analysis[i]['segment_id'], segment_analysis[i+1]['segment_id'])
        
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='darkblue')))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=0),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=400))
        
        st.plotly_chart(fig, use_container_width=True, key="segment_flow_graph")
        
        # 各セグメントの詳細
        st.subheader("各セグメントの詳細")
        
        for seg in segment_analysis:
            with st.expander(f"📌 セグメント {seg['segment_id']}: {seg['theme']}"):
                st.markdown(f"**要約:** {seg['summary']}")
                st.markdown(f"**総語数:** {seg['total_words']} | **ユニーク語数:** {seg['unique_words']}")
                
                st.markdown("**主要語（上位20語）:**")
                words_df = pd.DataFrame(seg['top_words'], columns=['語', '頻度'])
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig = px.bar(words_df.head(10), x='語', y='頻度', 
                                title=f'セグメント{seg["segment_id"]}の主要語')
                    st.plotly_chart(fig, use_container_width=True, key=f"segment_words_{seg['segment_id']}")
                
                with col2:
                    st.dataframe(words_df, use_container_width=True, height=400)
    else:
        st.info("まず「データ読み込み」タブで分析を実行してください。")

with tab4:
    st.header("🔄 語の遷移分析")
    
    if st.session_state.analyzed_data:
        data = st.session_state.analyzed_data
        transitions = data['transitions']
        segment_analysis = data['segment_analysis']
        
        st.subheader("セグメント間の語の引き継ぎ")
        
        # 遷移の可視化
        for trans in transitions:
            from_seg = segment_analysis[trans['from_segment'] - 1]
            to_seg = segment_analysis[trans['to_segment'] - 1]
            
            influence_pct = trans['influence_score'] * 100
            
            with st.expander(f"🔀 {from_seg['theme']} → {to_seg['theme']} （影響力: {influence_pct:.1f}%）"):
                st.markdown(f"**引き継がれた語:** {len(trans['common_words'])}語")
                
                if trans['common_words']:
                    st.markdown("**共通語:**")
                    st.write(", ".join(trans['common_words']))
                    
                    # 影響力メーター
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=influence_pct,
                        title={'text': "影響力スコア"},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "darkblue"},
                               'steps': [
                                   {'range': [0, 30], 'color': "lightgray"},
                                   {'range': [30, 70], 'color': "gray"},
                                   {'range': [70, 100], 'color': "lightblue"}],
                               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}}))
                    
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True, key=f"influence_gauge_{trans['from_segment']}_{trans['to_segment']}")
                else:
                    st.info("共通語が見つかりませんでした。テーマが大きく変化しています。")
        
        # 全体の遷移マトリックス
        st.subheader("遷移マトリックス")
        
        matrix_data = []
        for trans in transitions:
            matrix_data.append({
                'From': f"S{trans['from_segment']}",
                'To': f"S{trans['to_segment']}",
                'Score': trans['influence_score']
            })
        
        if matrix_data:
            matrix_df = pd.DataFrame(matrix_data)
            fig = px.bar(matrix_df, x='From', y='Score', color='To',
                        title='セグメント間の影響力スコア',
                        labels={'Score': '影響力', 'From': '元セグメント'})
            st.plotly_chart(fig, use_container_width=True, key="transition_matrix")
    else:
        st.info("まず「データ読み込み」タブで分析を実行してください。")

# フッター
st.markdown("---")
st.markdown("💡 **ヒント:** Groq APIは無料枠があります。[console.groq.com](https://console.groq.com)で登録してください。")
