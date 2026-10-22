import React, { useEffect, useState } from 'react';
import Antigravity from './Antigravity';
import './styles.css';

export default function App() {
  const [data, setData] = useState({
    title: 'Smart Parking Violation Detection',
    subtitle: 'AI-powered parking scene analysis',
    badge: 'Autonomous CV Lab',
  });

  useEffect(() => {
    const handleMessage = (event) => {
      if (event.data?.type === 'streamlit:render') {
        const args = event.data.args || {};
        setData({
          title: args.title || 'Smart Parking Violation Detection',
          subtitle: args.subtitle || 'AI-powered parking scene analysis',
          badge: args.badge || 'Autonomous CV Lab',
        });
      }
    };

    window.addEventListener('message', handleMessage);

    // Notify Streamlit that component is ready and set frame height to 400px
    window.parent.postMessage(
      { isStreamlitMessage: true, type: 'streamlit:componentReady', apiVersion: 1 },
      '*'
    );
    window.parent.postMessage(
      { isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height: 400 },
      '*'
    );

    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, []);

  return (
    <div className="antigravity-hero-wrapper" style={{ width: '100%', height: '400px', position: 'relative' }}>
      <Antigravity
        colors={['#5227FF', '#FF9FFC', '#B497CF']}
        count={1500}
        magnetRadius={9.5}
        ringRadius={4}
        particleSize={0.9}
        autoAnimate={false}
        particleShape="sphere"
        pulseSpeed={1.7}
      />
      <div className="antigravity-hero-overlay">
        <div className="antigravity-hero-content">
          {data.badge && (
            <div className="hero-badge">
              <span className="hero-badge-dot"></span>
              <span>{data.badge}</span>
            </div>
          )}
          <h1 className="hero-title">{data.title}</h1>
          <p className="hero-subtitle">{data.subtitle}</p>
        </div>
      </div>
    </div>
  );
}
