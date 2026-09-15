import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '25mb' }));

// Helper to get GoogleGenAI client
function getAIClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  return new GoogleGenAI({
    apiKey: apiKey || '',
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    store: 'Tawakkal Paint, Electronic & Senetry Store',
    timestamp: new Date().toISOString(),
  });
});

// System prompt for Tawakkal Store AI Assistant
const STORE_SYSTEM_INSTRUCTION = `
You are the official AI Technical Advisor and Customer Support Assistant for "Tawakkal Paint, Electronic & Senetry Store".
Your shop provides:
1. Electric & House Wiring: 
   - 100% Pure Copper single-core and multi-strand cables: 7/0.29 (2.5mm² equivalent, up to 15A), 3/0.29 (1.0mm² equivalent for LED lights & fans), 7/0.36 (4.0mm² heavy cable for 1.5/2.0 Ton AC & geysers), 1.5mm flexible wire.
   - Piano switch boards, modular sockets, SP/DP mini circuit breakers (20A, 32A, 63A), 12W SMD downlights, LED tube lights, PVC electrical tapes, digital voltage testers, fan capacitors.
   - Note: The store specializes in household and commercial wiring, fittings, switches, lighting, and protection. Big heavy industrial equipment/heavy generators are NOT carried (as specified by store owner).
2. Paints & Finishes (Behtareen / Premium variety):
   - Luxury Matt Velvet Plastic Emulsion (washable interior), Weather-Shield Anti-Fungal exterior paint (7-year protection), Super High-Gloss synthetic enamel for wooden doors & iron grills, ready-to-use acrylic wall putty (20kg buckets), primers/sealers, aerosol spray paints (400ml), roller trays, 1" to 4" pure bristle brushes, silicon carbide sandpapers.
3. Sanitary & Plumbing (Senetry):
   - PPRC hot & cold high-pressure pipes (25mm PN-20), brass female threaded elbows (25mm x 1/2"), PPRC stop valves, solid brass chrome bib cocks, stainless steel Muslim showers / bidets, flexible waste bottle traps, Teflon thread seal tape, anti-leak waterproof silicone sealants.
4. Hardware & Hand Tools:
   - 1000V safe VDE insulated electrician screwdrivers, 8-inch combination pliers, 5-meter auto-lock steel measuring tape, wall rawl plugs (gilli) and screws box.

Your demeanor:
- Helpful, polite, professional, and knowledgeable in local standards (Pakistan / South Asia & international).
- Respond in the language the user speaks: English, Urdu (اردو), or Roman Urdu.
- When advising on wiring, always emphasize electrical safety and proper gauge selection (e.g., 7/0.36 or 4mm² for 1.5 Ton AC, 7/0.29 for general plugs, 3/0.29 for lights).
- When advising on paint, provide clear coverage estimates (approx 140-160 sq.ft per gallon for emulsion) and prep steps (putty, sanding, primer).
- Keep answers practical, structured, and easy for non-experts to follow.
`;

// Gemini Multi-Turn Chat Endpoint
app.post('/api/gemini/chat', async (req, res) => {
  try {
    const { messages, highThinking, fastMode } = req.body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required' });
    }

    const ai = getAIClient();

    // Select model according to instructions:
    // gemini-3.1-pro-preview for complex tasks / high thinking
    // gemini-3.1-flash-lite for fast mode
    // gemini-3.5-flash for general tasks
    let modelName = 'gemini-3.5-flash';
    let config: any = {
      systemInstruction: STORE_SYSTEM_INSTRUCTION,
    };

    if (highThinking) {
      modelName = 'gemini-3.1-pro-preview';
      config.thinkingConfig = { thinkingLevel: ThinkingLevel.HIGH };
      // Note: Do not set maxOutputTokens when using thinking mode as per instructions
    } else if (fastMode) {
      modelName = 'gemini-3.1-flash-lite';
    }

    // Format conversation history for Gemini SDK
    const contents: any[] = messages.map((m: any) => ({
      role: m.sender === 'user' ? 'user' : 'model',
      parts: [{ text: m.text }],
    }));

    let response;
    try {
      response = await ai.models.generateContent({
        model: modelName,
        contents,
        config,
      });
    } catch (modelErr: any) {
      console.warn(`Primary model ${modelName} encountered error, falling back to gemini-3.8-flash:`, modelErr?.message);
      // Resilient fallback
      response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents,
        config: {
          systemInstruction: STORE_SYSTEM_INSTRUCTION,
        },
      });
      modelName = 'gemini-3.8-flash';
    }

    const replyText = response.text || 'I am ready to help you with wiring, paints, or sanitary fittings.';

    return res.json({
      text: replyText,
      modelUsed: modelName,
    });
  } catch (error: any) {
    console.error('Chat endpoint error:', error);
    return res.status(500).json({
      error: 'Failed to process AI chat inquiry',
      details: error?.message,
    });
  }
});

// Gemini Image Understanding Endpoint (Analyze broken part / paint color / hardware)
// Per requirement: "You MUST add image understanding to the app using model gemini-3.1-pro-preview"
app.post('/api/gemini/analyze-part', async (req, res) => {
  try {
    const { imageBase64, mimeType = 'image/jpeg', question } = req.body;

    if (!imageBase64) {
      return res.status(400).json({ error: 'imageBase64 is required' });
    }

    const ai = getAIClient();
    const cleanBase64 = imageBase64.replace(/^data:image\/[a-zA-Z0-9+]+;base64,/, '');

    const analysisPrompt = `
You are the Master Hardware, Electrical, Paint & Sanitary Specialist at "Tawakkal Paint, Electronic & Senetry Store".
Examine the uploaded image carefully. The customer is asking for assistance or replacement parts for their home/office maintenance.

Customer note/question: "${question || 'Identify this hardware part, electrical item, plumbing fitting, or paint color and recommend the exact store replacement.'}"

Perform a thorough visual diagnosis and return your analysis strictly as a JSON object with this exact JSON structure:
{
  "partName": "Name and specific specification of the identified item (e.g. 25mm PPRC Female Brass Elbow, 7/0.29 Copper Cable, Burnt 20A MCB, Brass Bib Cock, Off-White Matt Emulsion, etc.)",
  "detectedCategory": "One of: Electrical, Paints, Sanitary, Hardware",
  "condition": "Visual condition assessment (e.g. Leaking thread wear, burnt electrical terminal, peeling wall paint, worn gasket, new installation)",
  "replacementRecommendation": "Exact recommended replacement product or size available at Tawakkal Store",
  "suggestedProductId": "One of our catalog product IDs if matching: elec-001, elec-002, elec-003, elec-004, elec-005, elec-006, elec-007, elec-008, paint-001, paint-002, paint-003, paint-004, paint-005, paint-006, paint-007, san-001, san-002, san-003, san-004, san-005, san-006, hard-001, hard-002, hard-003, hard-004",
  "diySteps": [
    "Step 1 to safely replace or repair this item",
    "Step 2...",
    "Step 3..."
  ],
  "safetyTips": "Important safety precaution (e.g. Always switch off main DP breaker before touching wires, or turn off main overhead water valve before unthreading tap)",
  "toolsRequired": ["List of tools needed like 8-inch pliers, Teflon tape, screwdriver, etc."],
  "estimatedCostPKR": 1200
}

Return valid parseable JSON only. Do not wrap in markdown quotes if possible or standard \`\`\`json block.
`;

    let modelName = 'gemini-3.1-pro-preview';
    let response;

    try {
      response = await ai.models.generateContent({
        model: modelName,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: cleanBase64,
              },
            },
            {
              text: analysisPrompt,
            },
          ],
        },
        config: {
          responseMimeType: 'application/json',
        },
      });
    } catch (modelErr: any) {
      console.warn(`Primary model ${modelName} encountered error for image analysis, falling back to gemini-3.8-flash:`, modelErr?.message);
      modelName = 'gemini-3.8-flash';
      response = await ai.models.generateContent({
        model: modelName,
        contents: {
          parts: [
            {
              inlineData: {
                mimeType,
                data: cleanBase64,
              },
            },
            {
              text: analysisPrompt,
            },
          ],
        },
        config: {
          responseMimeType: 'application/json',
        },
      });
    }

    const responseText = response.text || '{}';
    let parsedData;
    try {
      // Clean possible backticks
      const cleanJson = responseText.replace(/```json\n?/, '').replace(/```\n?/, '').trim();
      parsedData = JSON.parse(cleanJson);
    } catch (jsonErr) {
      console.error('Failed to parse JSON from model response:', responseText);
      parsedData = {
        partName: 'Hardware / Sanitary Component',
        detectedCategory: 'Hardware',
        condition: 'Requires inspection',
        replacementRecommendation: 'Please visit Tawakkal Store or speak with our technician.',
        diySteps: ['Turn off main power/water supply before inspecting.', 'Bring sample to Tawakkal Store or compare specs.'],
        safetyTips: 'Always ensure personal safety and shut off main utilities.',
        toolsRequired: ['Combination Pliers', 'Screwdriver', 'Measuring Tape'],
        estimatedCostPKR: 850,
      };
    }

    return res.json({
      success: true,
      analysis: parsedData,
      modelUsed: modelName,
    });
  } catch (error: any) {
    console.error('Image analysis error:', error);
    return res.status(500).json({
      error: 'Failed to analyze part image',
      details: error?.message,
    });
  }
});

// Paint Gallon & Wiring Load Estimator with High Thinking
app.post('/api/gemini/calculate-estimate', async (req, res) => {
  try {
    const { type, inputs } = req.body;
    const ai = getAIClient();

    let calculationPrompt = '';
    if (type === 'paint') {
      calculationPrompt = `
You are the senior Paint Estimator at Tawakkal Paint Store.
Calculate required paint quantities for the following room/surface dimensions:
Room Length: ${inputs.length || 14} ft
Room Width: ${inputs.width || 12} ft
Room Height: ${inputs.height || 10} ft
Doors: ${inputs.doors || 1}
Windows: ${inputs.windows || 2}
Surface condition: ${inputs.surfaceCondition || 'New Plaster requiring primer + 2 coats of emulsion'}

Provide accurate calculation of:
1. Total Wall Surface Area (sq.ft) excluding door/window deductions
2. Ceiling Area (sq.ft)
3. Gallons of Wall Putty required for smooth surface
4. Gallons of Primer/Sealer required
5. Gallons of Luxury Matt Plastic Emulsion required for 2 coats
6. Practical painter advice (dilution with clean water, roller nap, sanding grit P120/P240).
`;
    } else {
      calculationPrompt = `
You are the Electrical Engineer at Tawakkal Electronic Store.
Calculate safe electrical wire gauge, breaker sizing, and piping conduit for the following load:
Appliance / Load description: ${inputs.appliances || '1.5 Ton Inverter AC + 1 Refrigerator + 6 LED ceiling downlights + 2 Ceiling Fans'}
Distance to Distribution Board: ${inputs.distance || 45} ft

Provide:
1. Total Estimated Running Watts & Amps
2. Recommended Cable Specification (e.g. 7/0.36 or 7/0.29 Pure Copper)
3. Recommended Circuit Breaker (MCB) rating (e.g. 20A or 32A C-Curve)
4. Earth wire recommendation (e.g. 3/0.29 Green copper)
5. Crucial electrical safety rules to avoid fire hazards.
`;
    }

    // Using gemini-3.1-pro-preview with ThinkingLevel.HIGH as mandated
    let modelName = 'gemini-3.1-pro-preview';
    let response;
    try {
      response = await ai.models.generateContent({
        model: modelName,
        contents: calculationPrompt,
        config: {
          thinkingConfig: { thinkingLevel: ThinkingLevel.HIGH },
          systemInstruction: 'You are an authoritative engineering and paint calculation specialist.',
        },
      });
    } catch (err: any) {
      console.warn('Fallback from 3.1-pro to 3.8-flash in calculation endpoint:', err?.message);
      modelName = 'gemini-3.8-flash';
      response = await ai.models.generateContent({
        model: modelName,
        contents: calculationPrompt,
      });
    }

    return res.json({
      success: true,
      result: response.text,
      modelUsed: modelName,
    });
  } catch (error: any) {
    console.error('Calculation error:', error);
    return res.status(500).json({ error: error?.message });
  }
});

// Vite Middleware & Static Serving
async function setupServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Tawakkal Store Server running on http://0.0.0.0:${PORT}`);
  });
}

setupServer();
