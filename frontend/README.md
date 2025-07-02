# Medical Symptom Bot Frontend

A modern, responsive React frontend for the AI-powered medical symptom collection system.

## Features

- 🎨 **Modern UI/UX**: Clean, professional design with smooth animations
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 💬 **Real-time Chat**: Interactive conversation interface with typing indicators
- 🏥 **Patient Information**: Comprehensive form for collecting consultation details
- 🔄 **State Management**: Efficient state handling with React hooks
- 🎯 **Context-Aware**: Dynamic questioning based on patient demographics
- 🔒 **Secure**: HTTPS-ready with proper error handling

## Tech Stack

- **React 18** - Modern React with hooks
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations and transitions
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client for API communication
- **React Router** - Client-side routing
- **React Hot Toast** - Toast notifications

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Backend Flask API running on `http://localhost:5000`

### Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

4. **Open your browser:**
   Navigate to `http://localhost:3000`

### Building for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Header.js
│   │   ├── MessageBubble.js
│   │   ├── PatientInfoForm.js
│   │   └── TypingIndicator.js
│   ├── pages/
│   │   ├── Home.js
│   │   └── Chat.js
│   ├── App.js
│   ├── index.js
│   └── index.css
├── package.json
├── tailwind.config.js
└── README.md
```

## API Integration

The frontend communicates with your Flask backend through the following endpoints:

- `POST /start_conversation` - Initialize a new conversation
- `POST /message` - Send a message and receive bot response

### Configuration

The API base URL is configured in `package.json` with the proxy setting:
```json
{
  "proxy": "http://localhost:5000"
}
```

## Key Components

### Home Page (`/`)
- Hero section with call-to-action
- Feature highlights
- Benefits overview
- Professional medical branding

### Chat Page (`/chat`)
- Patient information form
- Real-time conversation interface
- Message history with timestamps
- Typing indicators
- Responsive design

### Patient Information Form
- Doctor and clinic details
- Consultation type selection
- Medical specialty options
- Age group and gender selection
- Form validation

## Styling

The app uses Tailwind CSS with custom components:

- **Primary Colors**: Blue theme for medical professionalism
- **Components**: Pre-built button, input, and card styles
- **Animations**: Smooth transitions and micro-interactions
- **Responsive**: Mobile-first design approach

## Development

### Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

### Customization

1. **Colors**: Modify `tailwind.config.js` for brand colors
2. **Components**: Update component styles in `src/index.css`
3. **API**: Adjust endpoints in the Chat component
4. **Animations**: Customize Framer Motion animations

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Medical Symptom Bot system. 