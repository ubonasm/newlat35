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

# Groq API (Free tier available)
import requests

st.set_page_config(page_title="new LAT35: Lesson Analysis Application", layout="wide", page_icon="📚")

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = None
if 'segments' not in st.session_state:
    st.session_state.segments = None

class ClassroomAnalyzer:
    def __init__(self, custom_dict=None, pos_filter=None):
        self.tokenizer = Tokenizer()
        self.custom_dict = custom_dict or {}
        self.pos_filter = pos_filter or ['名詞', '動詞', '形容詞']
        
    def load_custom_dictionary(self, dict_df):
        """Load user dictionary"""
        custom_dict = {}
        for _, row in dict_df.iterrows():
            word = str(row.iloc[0]).strip()
            reading = str(row.iloc[1]).strip() if len(row) > 1 else word
            custom_dict[word] = reading
        return custom_dict
    
    def tokenize_with_custom_dict(self, text):
        """Morphological analysis with custom dictionary priority"""
        tokens = []
        remaining_text = text
        
        # Detect custom dictionary words first
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
        
        # Analyze remaining text with standard morphological analysis
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
    
    def extract_keywords(self, tokens):
        """Extract keywords based on selected POS"""
        keywords = []
        for t in tokens:
            # Always include custom dictionary words
            if t['pos'] == 'カスタム':
                keywords.append(t['surface'])
            # Check if POS is in filter
            elif t['pos'] in self.pos_filter:
                keywords.append(t['surface'])
        return keywords
    
    def analyze_speakers(self, df):
        """Analyze utterances by speaker"""
        speaker_analysis = defaultdict(lambda: {'utterances': [], 'word_freq': Counter()})
        
        for _, row in df.iterrows():
            speaker = str(row['Speaker']).strip()
            utterance = str(row['Utterance']).strip()
            
            tokens = self.tokenize_with_custom_dict(utterance)
            words = self.extract_keywords(tokens)
            
            speaker_analysis[speaker]['utterances'].append(utterance)
            speaker_analysis[speaker]['word_freq'].update(words)
        
        return speaker_analysis
    
    def analyze_speaker_claims(self, speaker, utterances, groq_api_key):
        """Analyze speaker's claims and tendencies using AI"""
        utterances_text = "\n".join([f"- {u}" for u in utterances[:20]])
        
        prompt = f"""Analyze the following utterances by speaker "{speaker}" and provide:
1. Main claims or positions (what they argue or advocate for)
2. Overall tendency of their speech (teaching style, questioning pattern, etc.)

Keep the analysis concise (2-3 sentences each) and write in English.

Utterances:
{utterances_text}

Provide the response in JSON format:
{{"claims": "...", "tendency": "..."}}
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
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return analysis
        except Exception as e:
            pass
        
        return {
            "claims": "Analysis unavailable",
            "tendency": "Analysis unavailable"
        }
    
    def segment_classroom(self, df, groq_api_key):
        """Segment classroom transcript using AI"""
        full_text = ""
        for _, row in df.iterrows():
            full_text += f"[{row['No']}] {row['Speaker']}: {row['Utterance']}\n"
        
        prompt = f"""この授業記録を分析し、テーマや内容の変化に基づいて3〜7個の意味のあるセグメントに分割してください。

各セグメントについて以下を提供してください：
- segment_id（1から始まる番号）
- start_no（セグメントが始まる発言番号）
- end_no（セグメントが終わる発言番号）
- theme（授業記録の実際の言葉を使った説明的な名前、最大30文字）
- summary（簡潔な説明、最大60文字）

重要：themeとsummaryは、授業記録に実際に出てくる言葉やフレーズを使って、日本語で記述してください。

授業記録:
{full_text[:5000]}

以下の正確なJSON形式でのみ返してください：
{{"segments": [{{"segment_id": 1, "start_no": 1, "end_no": 15, "theme": "導入と目標の確認", "summary": "授業の目標を説明"}}]}}
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
                    "temperature": 0.5,
                    "max_tokens": 2000
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    segments_data = json.loads(json_match.group())
                    segments = segments_data.get('segments', [])
                    
                    if segments and len(segments) > 0:
                        return segments
            
        except Exception as e:
            st.warning(f"AI segmentation failed: {e}. Using automatic segmentation.")
        
        # Fallback: automatic segmentation with meaningful structure
        total_utterances = len(df)
        num_segments = min(5, max(3, total_utterances // 20))
        segment_size = total_utterances // num_segments
        segments = []
        
        segment_names = ['導入', '展開', '練習', '議論', 'まとめ']
        segment_summaries = ['授業の導入部分', '内容の展開', '練習活動', '議論と考察', 'まとめと振り返り']
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = total_utterances - 1 if i == num_segments - 1 else (i + 1) * segment_size - 1
            
            segments.append({
                'segment_id': i + 1,
                'start_no': int(df.iloc[start_idx]['No']),
                'end_no': int(df.iloc[end_idx]['No']),
                'theme': segment_names[i] if i < len(segment_names) else f'フェーズ{i + 1}',
                'summary': segment_summaries[i] if i < len(segment_summaries) else f'授業フェーズ{i + 1}'
            })
        
        return segments
    
    def analyze_segments(self, df, segments):
        segment_analysis = []
        
        for seg in segments:
            seg_df = df[(df['No'] >= seg['start_no']) & (df['No'] <= seg['end_no'])]
            
            all_words = []
            for _, row in seg_df.iterrows():
                tokens = self.tokenize_with_custom_dict(str(row['Utterance']))
                words = [w for w in self.extract_keywords(tokens) if len(w) > 1]
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
        """Analyze word transitions between segments"""
        transitions = []
        
        for i in range(len(segment_analysis) - 1):
            current_seg = segment_analysis[i]
            next_seg = segment_analysis[i + 1]
            
            current_words = set([w[0] for w in current_seg['top_words'][:15]])
            next_words = set([w[0] for w in next_seg['top_words'][:15]])
            
            common_words = current_words & next_words
            
            if len(current_words) > 0:
                influence_score = len(common_words) / len(current_words)
            else:
                influence_score = 0
            
            transitions.append({
                'from_segment': current_seg['segment_id'],
                'to_segment': next_seg['segment_id'],
                'common_words': sorted(list(common_words)),
                'influence_score': influence_score
            })
        
        return transitions

# Streamlit UI
st.title("📚 new LAT35: Lesson Analysis Application")
st.markdown("Lesson transcript analysis tool using morphological analysis and AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    groq_api_key = st.text_input(
        "Groq API Key (Free)",
        type="password",
        help="Get your API key at https://console.groq.com (free tier available)"
    )
    
    st.markdown("---")
    
    st.subheader("Part-of-Speech Filter")
    st.markdown("Select which parts of speech to include in analysis:")
    
    pos_options = {
        'Nouns (名詞)': '名詞',
        'Verbs (動詞)': '動詞',
        'Adjectives (形容詞)': '形容詞',
        'Adverbs (副詞)': '副詞',
        'Others (その他)': 'その他'
    }
    
    selected_pos_labels = st.multiselect(
        "Select POS to analyze:",
        options=list(pos_options.keys()),
        default=['Nouns (名詞)', 'Verbs (動詞)', 'Adjectives (形容詞)'],
        help="Choose which parts of speech to include in keyword extraction"
    )
    
    selected_pos = [pos_options[label] for label in selected_pos_labels]
    
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. Enter your Groq API key
    2. Upload lesson transcript CSV
    3. (Optional) Upload custom dictionary
    4. Select parts of speech to analyze
    5. Run analysis
    """)

# Main area
tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Loading", "👥 Speaker Analysis", "📊 Segment Analysis", "🔄 Word Transitions"])

with tab1:
    st.header("Data Loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transcript File")
        classroom_file = st.file_uploader(
            "Upload CSV file (No, Speaker, Utterance)",
            type=['csv'],
            key='classroom'
        )
        
        if classroom_file:
            try:
                df = pd.read_csv(classroom_file)
                st.success(f"✅ Loaded {len(df)} rows of data")
                st.dataframe(df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
                df = None
    
    with col2:
        st.subheader("Custom Dictionary (Optional)")
        dict_file = st.file_uploader(
            "Upload CSV file (Word, Reading)",
            type=['csv'],
            key='dictionary'
        )
        
        custom_dict_df = None
        if dict_file:
            try:
                custom_dict_df = pd.read_csv(dict_file, header=None)
                st.success(f"✅ Loaded dictionary with {len(custom_dict_df)} words")
                st.dataframe(custom_dict_df.head(10), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        if not groq_api_key:
            st.error("Please enter your Groq API key")
        elif classroom_file is None:
            st.error("Please upload a transcript file")
        elif not selected_pos:
            st.error("Please select at least one part of speech")
        else:
            with st.spinner("Analyzing..."):
                analyzer = ClassroomAnalyzer(pos_filter=selected_pos)
                
                if custom_dict_df is not None:
                    analyzer.custom_dict = analyzer.load_custom_dictionary(custom_dict_df)
                    st.info(f"Custom dictionary: Applied {len(analyzer.custom_dict)} words")
                
                speaker_analysis = analyzer.analyze_speakers(df)
                
                st.info("Analyzing speaker claims and tendencies...")
                for speaker in speaker_analysis.keys():
                    utterances = speaker_analysis[speaker]['utterances']
                    claims_analysis = analyzer.analyze_speaker_claims(speaker, utterances, groq_api_key)
                    speaker_analysis[speaker]['claims'] = claims_analysis.get('claims', 'Analysis unavailable')
                    speaker_analysis[speaker]['tendency'] = claims_analysis.get('tendency', 'Analysis unavailable')
                
                segments = analyzer.segment_classroom(df, groq_api_key)
                
                segment_analysis = analyzer.analyze_segments(df, segments)
                
                transitions = analyzer.analyze_word_transitions(df, segments, segment_analysis)
                
                st.session_state.analyzed_data = {
                    'df': df,
                    'speaker_analysis': speaker_analysis,
                    'segments': segments,
                    'segment_analysis': segment_analysis,
                    'transitions': transitions,
                    'analyzer': analyzer,
                    'pos_filter': selected_pos
                }
                
                st.success("✅ Analysis complete! Check other tabs for results.")

with tab2:
    st.header("👥 Speaker Analysis")
    
    if st.session_state.analyzed_data:
        data = st.session_state.analyzed_data
        speaker_analysis = data['speaker_analysis']
        
        st.info(f"**Analyzing POS:** {', '.join(data.get('pos_filter', ['名詞', '動詞', '形容詞']))}")
        
        st.subheader("Claims and Characteristics by Speaker")
        
        sorted_speakers = sorted(
            speaker_analysis.items(), 
            key=lambda x: len(x[1]['utterances']), 
            reverse=True
        )
        
        for speaker, info in sorted_speakers:
            with st.expander(f"🗣️ {speaker} (Utterances: {len(info['utterances'])})"):
                st.markdown("### 📋 Claims and Positions")
                st.markdown(info.get('claims', 'Analysis unavailable'))
                
                st.markdown("### 📊 Overall Tendency")
                st.markdown(info.get('tendency', 'Analysis unavailable'))
                
                st.markdown("---")
                
                st.markdown("### 🔑 Key Words")
                top_words = info['word_freq'].most_common(15)
                
                if top_words:
                    words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                    fig = px.bar(words_df, x='Word', y='Frequency', title=f'{speaker}\'s Key Words')
                    st.plotly_chart(fig, use_container_width=True, key=f"speaker_chart_{speaker}")
                
                st.markdown("### 💬 Sample Utterances")
                for i, utterance in enumerate(info['utterances'][:3], 1):
                    st.markdown(f"{i}. {utterance}")
    else:
        st.info("Please run analysis in the 'Data Loading' tab first.")

with tab3:
    st.header("📊 Segment Analysis")
    
    if st.session_state.analyzed_data:
        data = st.session_state.analyzed_data
        segments = data['segments']
        segment_analysis = data['segment_analysis']
        
        st.info(f"**Analyzing POS:** {', '.join(data.get('pos_filter', ['名詞', '動詞', '形容詞']))}")
        
        st.subheader("Segment Flow")
        
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
        
        st.subheader("Segment Details")
        
        for i, seg in enumerate(segment_analysis):
            seg_info = segments[i]
            start_no = seg_info['start_no']
            end_no = seg_info['end_no']
            utterance_range = "No." + str(start_no) + " to No." + str(end_no)
            segment_id = seg['segment_id']
            expander_title = "📌 Segment " + str(segment_id) + ": " + seg['theme'] + " (" + utterance_range + ")"
            
            with st.expander(expander_title):
                st.markdown("**Utterance Range:** " + utterance_range)
                st.markdown("**Summary:** " + seg['summary'])
                st.markdown("**Total Words:** " + str(seg['total_words']) + " | **Unique Words:** " + str(seg['unique_words']))
                
                st.markdown("**Key Words (Top 20):**")
                words_df = pd.DataFrame(seg['top_words'], columns=['Word', 'Frequency'])
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    chart_title = "Segment " + str(segment_id) + " Key Words"
                    chart_key = "segment_words_" + str(segment_id)
                    fig = px.bar(words_df.head(10), x='Word', y='Frequency', title=chart_title)
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
                with col2:
                    st.dataframe(words_df, use_container_width=True, height=400)
    else:
        st.info("Please run analysis in the 'Data Loading' tab first.")

with tab4:
    st.header("🔄 Word Transition Analysis")
    
    if st.session_state.analyzed_data:
        data = st.session_state.analyzed_data
        transitions = data['transitions']
        segment_analysis = data['segment_analysis']
        
        st.info(f"**Analyzing POS:** {', '.join(data.get('pos_filter', ['名詞', '動詞', '形容詞']))}")
        
        st.subheader("Word Carryover Between Segments")
        
        for trans in transitions:
            from_seg = segment_analysis[trans['from_segment'] - 1]
            to_seg = segment_analysis[trans['to_segment'] - 1]
            
            influence_pct = trans['influence_score'] * 100
            
            with st.expander(f"🔀 {from_seg['theme']} → {to_seg['theme']} (Influence: {influence_pct:.1f}%)"):
                st.markdown(f"**Carried Over Words:** {len(trans['common_words'])} words")
                
                if trans['common_words']:
                    st.markdown("**Common Words:**")
                    st.write(", ".join(trans['common_words']))
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=influence_pct,
                        title={'text': "Influence Score"},
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
                    st.info("No common words found. Theme has changed significantly.")
        
        st.subheader("Transition Matrix")
        
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
                        title='Influence Score Between Segments',
                        labels={'Score': 'Influence', 'From': 'Source Segment'})
            st.plotly_chart(fig, use_container_width=True, key="transition_matrix")
    else:
        st.info("Please run analysis in the 'Data Loading' tab first.")

st.markdown("---")
st.markdown("💡 **Tip:** Groq API has a free tier. Register at [console.groq.com](https://console.groq.com).")
