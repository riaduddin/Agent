const io = require('socket.io-client');

const SERVER_URL = 'http://127.0.0.1:8060';
const TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJzdWIiOiI2ODgwODdlMzA1MTk0OTc2YWVhOTk0MDMiLCJlbWFpbCI6InJycmlhZHVkZGluQGdtYWlsLmNvbSIsInBhY2thZ2UiOiJ1bmxpbWl0ZWQiLCJpc192ZXJpZmllZCI6dHJ1ZSwicm9sZSI6InVzZXIiLCJpYXQiOjE3NjEzNzczMzAsImV4cCI6MTc2MTQ2MzczMH0.dUeYJu4jNbaSfN8jRloiFiRTie1WkJ1prWtMe-_RQLg';
const MESSAGE = 'Create a presentation about AI in Healthcare';

async function createPresentation() {
    const response = await fetch(`${SERVER_URL}/create-presentation`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${TOKEN}`
        },
        body: JSON.stringify({
            message: MESSAGE,
            file_urls: []
        })
    });
    const data = await response.json();
    return data.p_id;
}

async function startPresentation(p_id) {
    const response = await fetch(`${SERVER_URL}/start-presentation/${p_id}`, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${TOKEN}`
        }
    });
    return response.ok;
}

async function main() {
    console.log('🧪 Client Test Starting...');
    
    // Create presentation
    const p_id = await createPresentation();
    console.log(`📝 Created presentation: ${p_id}`);
    
    // Connect Socket.IO
    const socket = io(SERVER_URL, {
        query: {
            p_id: p_id,
            token: TOKEN
        }
    });
    
    socket.on('connect', () => {
        console.log('✅ Connected to Socket.IO');
    });
    
    socket.on('agent_output', (data) => {
        console.log('📨 Agent Output:', data);
    });
    
    socket.on('message', (data) => {
        console.log('📨 Message:', data);
    });
    
    // Start presentation
    await startPresentation(p_id);
    console.log('🚀 Started presentation');
    
    // Wait for events
    await new Promise(resolve => setTimeout(resolve, 30000));
    
    // Disconnect
    socket.disconnect();
    console.log('✅ Test completed');
}

main().catch(console.error);
