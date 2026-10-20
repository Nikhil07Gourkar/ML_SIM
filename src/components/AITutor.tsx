import React, { useState, useEffect, useRef } from 'react';
import {
  Bot,
  Send,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Sparkles,
  HelpCircle,
  Lightbulb,
  Code,
  RotateCcw,
  User,
} from 'lucide-react';

interface AITutorProps {
  context?: {
    lessonTitle?: string;
    algorithmId?: string;
    datasetName?: string;
    taskType?: string;
    metrics?: any;
    hyperparameters?: any;
  };
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'tutor';
  text: string;
  timestamp: string;
}

export const AITutor: React.FC<AITutorProps> = ({ context }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'msg_welcome',
      sender: 'tutor',
      text: `Hello! I'm your **AI Machine Learning Professor & Tutor**.
I can help explain machine learning algorithms from absolute scratch, diagnose your model's accuracy or error scores, suggest the best hyperparameters to try next, or write clean scikit-learn code!

What concept or simulation would you like to explore today?`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);

  const [inputQuestion, setInputQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Handle Speech-to-Text via Web Speech API
  const toggleSpeechRecognition = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Please type your question.');
      return;
    }

    if (isListening) {
      setIsListening(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-US';
      recognition.interimResults = false;
      recognition.maxAlternatives = 1;

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputQuestion((prev) => (prev ? `${prev} ${transcript}` : transcript));
        setIsListening(false);
      };

      recognition.onerror = () => {
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognition.start();
    } catch (err) {
      setIsListening(false);
    }
  };

  // Handle Text-to-Speech via Web Speech API
  const speakText = (text: string) => {
    if (!('speechSynthesis' in window)) return;

    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    // Strip markdown formatting for cleaner speech
    const cleanText = text.replace(/[*_#`[\]()]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  };

  // Send message to server /api/tutor
  const handleSendMessage = async (textToSend?: string) => {
    const question = (textToSend || inputQuestion).trim();
    if (!question || isLoading) return;

    const userMsg: ChatMessage = {
      id: `usr_${Date.now()}`,
      sender: 'user',
      text: question,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuestion('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/tutor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, context }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      const tutorMsg: ChatMessage = {
        id: `tut_${Date.now()}`,
        sender: 'tutor',
        text: data.reply || 'Here is an explanation of the topic.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, tutorMsg]);
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: `err_${Date.now()}`,
        sender: 'tutor',
        text: `### Machine Learning Concept Explanation
Based on your current workspace:
- **Algorithm**: ${context?.algorithmId || 'General Machine Learning'}
- **Dataset**: ${context?.datasetName || 'Active Dataset'}

Remember that machine learning algorithms learn by adjusting parameters to minimize a mathematical loss function. For regression, we minimize Mean Squared Error; for classification, we minimize Cross-Entropy.
Try adjusting your hyperparameters or checking for feature scaling!`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  // Preset quick questions
  const quickQuestions = [
    'Why is my accuracy low?',
    'Explain the bias-variance tradeoff simply',
    'How do I choose K in K-Means clustering?',
    'What is data leakage and how do I prevent it?',
  ];

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 text-xs font-medium text-indigo-400">
            <Bot className="w-4 h-4" />
            <span>Context-Aware AI Tutor (Gemini 3.8 Flash)</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Machine Learning AI Tutor
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Ask any question about algorithms, mathematical formulas, evaluation scores, or Python code.
          </p>
        </div>

        {/* Current Context Pill Bar */}
        {context?.algorithmId && (
          <div className="p-2.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300 flex items-center gap-2">
            <span className="text-indigo-400 font-semibold uppercase">{context.algorithmId}</span>
            <span className="text-slate-600">·</span>
            <span>{context.datasetName}</span>
          </div>
        )}
      </div>

      {/* Chat Messages Log */}
      <div className="rounded-2xl bg-slate-900 border border-slate-800 h-[480px] p-6 overflow-y-auto space-y-5">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex items-start gap-3.5 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.sender === 'tutor' && (
              <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/40 flex items-center justify-center text-indigo-400 shrink-0">
                <Bot className="w-4 h-4" />
              </div>
            )}

            <div
              className={`max-w-2xl rounded-2xl p-4 text-xs md:text-sm leading-relaxed space-y-2 ${
                msg.sender === 'user'
                  ? 'bg-indigo-600 text-white rounded-tr-none'
                  : 'bg-slate-950 border border-slate-800/80 text-slate-200 rounded-tl-none'
              }`}
            >
              <div className="whitespace-pre-wrap font-sans">
                {msg.text}
              </div>

              <div className="flex items-center justify-between pt-1 border-t border-slate-800/40 text-[10px] text-slate-500">
                <span>{msg.timestamp}</span>
                {msg.sender === 'tutor' && (
                  <button
                    onClick={() => speakText(msg.text)}
                    title="Read response aloud"
                    className="hover:text-indigo-400 transition-colors flex items-center gap-1"
                  >
                    {isSpeaking ? <VolumeX className="w-3.5 h-3.5 text-indigo-400" /> : <Volume2 className="w-3.5 h-3.5" />}
                    <span>Listen</span>
                  </button>
                )}
              </div>
            </div>

            {msg.sender === 'user' && (
              <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center text-slate-300 shrink-0">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex items-center gap-3 text-xs text-indigo-400 font-medium animate-pulse">
            <Bot className="w-5 h-5" />
            <span>AI Tutor is formulating an explanation...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested Quick Prompt Buttons */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-slate-500">Suggested:</span>
        {quickQuestions.map((q, idx) => (
          <button
            key={idx}
            onClick={() => handleSendMessage(q)}
            className="px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-white text-xs transition-colors"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Input Field with Speech Recognition */}
      <div className="relative flex items-center gap-2">
        <div className="relative flex-1">
          <input
            type="text"
            value={inputQuestion}
            onChange={(e) => setInputQuestion(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Ask a question about machine learning, formulas, or results..."
            className="w-full bg-slate-900 border border-slate-800 rounded-xl pl-4 pr-12 py-3 text-xs md:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
          />

          <button
            onClick={toggleSpeechRecognition}
            title={isListening ? 'Stop Listening' : 'Voice Input (Microphone)'}
            className={`absolute right-3 top-3 p-1 rounded-md transition-colors ${
              isListening ? 'text-rose-400 animate-pulse bg-rose-500/10' : 'text-slate-400 hover:text-white'
            }`}
          >
            {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
          </button>
        </div>

        <button
          disabled={!inputQuestion.trim() || isLoading}
          onClick={() => handleSendMessage()}
          className="p-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded-xl shadow-sm shadow-indigo-600/30 transition-all flex items-center justify-center shrink-0"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
