import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, User, Bot, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';
import axios from 'axios';
import PatientInfoForm from '../components/PatientInfoForm';
import MessageBubble from '../components/MessageBubble';
import TypingIndicator from '../components/TypingIndicator';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationStarted, setConversationStarted] = useState(false);
  const [patientInfo, setPatientInfo] = useState(null);
  const [threadId, setThreadId] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const startConversation = async (info) => {
    try {
      setIsLoading(true);
      const response = await axios.post('/start_conversation', {
        thread_id: `thread_${Date.now()}`,
        doctor_name: info.doctor_name,
        consultation_type: info.consultation_type,
        specialty: info.specialty,
        age_group: info.age_group,
        gender: info.gender,
        clinic_name: info.clinic_name
      });

      setThreadId(response.data.thread_id);
      setPatientInfo(info);
      setConversationStarted(true);
      
      // Add welcome message
      setMessages([{
        id: Date.now(),
        type: 'bot',
        content: `Hello! I'm your medical assistant. I'll help you with symptom assessment. How can I assist you today?`,
        timestamp: new Date()
      }]);

      toast.success('Conversation started successfully!');
    } catch (error) {
      console.error('Error starting conversation:', error);
      toast.error('Failed to start conversation. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post('/message', {
        thread_id: threadId,
        message: inputMessage,
        age_group: patientInfo?.age_group,
        gender: patientInfo?.gender,
        specialty: patientInfo?.specialty
      });

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.data.reply,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Failed to send message. Please try again.');
      
      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (!conversationStarted) {
    return (
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-4">
              Start Your Medical Consultation
            </h1>
            <p className="text-lg text-gray-600">
              Please provide your information to begin the symptom assessment
            </p>
          </div>
          <PatientInfoForm onSubmit={startConversation} isLoading={isLoading} />
        </motion.div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto h-[calc(100vh-200px)] flex flex-col">
      {/* Chat Header */}
      <div className="bg-white rounded-t-xl border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary-100 rounded-full flex items-center justify-center">
              <Bot className="w-5 h-5 text-primary-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">Medical Assistant</h2>
              <p className="text-sm text-gray-500">
                {patientInfo?.specialty} • {patientInfo?.age_group} • {patientInfo?.gender}
              </p>
            </div>
          </div>
          <div className="text-sm text-gray-500">
            Thread: {threadId?.slice(-8)}
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 bg-gray-50 overflow-y-auto p-4 space-y-4">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              <MessageBubble message={message} />
            </motion.div>
          ))}
        </AnimatePresence>
        
        {isLoading && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="bg-white rounded-b-xl border-t border-gray-200 p-4">
        <div className="flex space-x-3">
          <div className="flex-1">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Describe your symptoms..."
              className="input-field resize-none h-12 py-3"
              rows="1"
              disabled={isLoading}
            />
          </div>
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed h-12 px-4"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chat; 