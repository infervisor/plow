"""TTS prompt sets of the repo harnesses (scripts/tts/{veena_ref,chatterbox_ref,mtl_prompts}.py), as used
for BASELINE.md and the recipe gates."""

# (voice, text)
VEENA_PROMPTS = [('kavya',
  'आज मैंने एक नई तकनीक के बारे में सीखा जो कृत्रिम बुद्धिमत्ता का उपयोग करके मानव जैसी आवाज़ उत्पन्न कर '
  'सकती है।'),
 ('agastya',
  'Today I learned about a new technology that uses artificial intelligence to generate human-like voices.'),
 ('maitri', 'मैं तो पूरा presentation prepare कर चुका हूं! कल रात को ही मैंने पूरा code base चेक किया।'),
 ('vinaya',
  'The quick brown fox jumps over the lazy dog, and then it takes a long nap in the warm afternoon sun.'),
 ('kavya', 'Hello, how are you doing today?'),
 ('agastya', 'नमस्ते, आप कैसे हैं? मुझे आशा है कि आपका दिन अच्छा गुजर रहा है।'),
 ('maitri', 'Please confirm your appointment for tomorrow at three thirty in the afternoon.'),
 ('vinaya', 'भारत एक विशाल और विविधताओं से भरा हुआ देश है जहाँ अनेक भाषाएँ बोली जाती हैं।')]

CBX_TEXTS = ['Today I learned about a new technology that uses artificial intelligence to generate human-like voices.',
 'The quick brown fox jumps over the lazy dog, and then it takes a long nap in the warm afternoon sun.',
 'Hello, how are you doing today?',
 'Please confirm your appointment for tomorrow at three thirty in the afternoon.',
 'Streaming speech synthesis needs both low latency for the first chunk and high throughput under load.',
 'It was a bright cold day in April, and the clocks were striking thirteen.',
 'Your package has shipped and should arrive within two business days.',
 'Machine learning systems are only as good as the data they are trained on.']

# (language, text)
MTL_PROMPTS = [('en', 'Thanks for calling, how can I help you today?'),
 ('hi', 'कॉल करने के लिए धन्यवाद, आज मैं आपकी क्या मदद कर सकता हूँ?'),
 ('zh', '感谢您的来电，今天有什么可以帮您的吗？'),
 ('ja', 'お電話ありがとうございます。本日はどのようなご用件でしょうか？'),
 ('es', 'Gracias por llamar, ¿en qué puedo ayudarle hoy?'),
 ('fr', "Merci de votre appel, comment puis-je vous aider aujourd'hui ?"),
 ('ar', 'شكرا لاتصالك، كيف يمكنني مساعدتك اليوم؟'),
 ('de', 'Danke für Ihren Anruf, wie kann ich Ihnen heute helfen?'),
 ('en', 'Your appointment is confirmed for Tuesday at three thirty in the afternoon.'),
 ('hi', 'आपकी अपॉइंटमेंट मंगलवार दोपहर साढ़े तीन बजे के लिए पक्की हो गई है।'),
 ('zh', '您的预约已确认，时间是星期二下午三点半。'),
 ('ja', 'ご予約は火曜日の午後三時半に確定しました。'),
 ('es', 'Su cita está confirmada para el martes a las tres y media de la tarde.'),
 ('fr', 'Votre rendez-vous est confirmé pour mardi à quinze heures trente.'),
 ('ar', 'تم تأكيد موعدك يوم الثلاثاء الساعة الثالثة والنصف بعد الظهر.'),
 ('de', 'Ihr Termin ist für Dienstag um halb vier am Nachmittag bestätigt.'),
 ('en', 'I can transfer you to a specialist, or we can reschedule the delivery for tomorrow morning.'),
 ('hi', 'मैं आपको किसी विशेषज्ञ से जोड़ सकता हूँ, या हम डिलीवरी कल सुबह के लिए तय कर सकते हैं।'),
 ('zh', '我可以帮您转接专员，或者我们可以把送货改到明天上午。'),
 ('ja', '担当者におつなぎするか、配達を明日の午前中に変更することができます。'),
 ('es',
  'Puedo transferirle con un especialista, o podemos reprogramar la entrega para mañana por la mañana.'),
 ('fr', 'Je peux vous transférer à un spécialiste, ou nous pouvons reporter la livraison à demain matin.'),
 ('ar', 'يمكنني تحويلك إلى أحد المختصين، أو يمكننا تأجيل التوصيل إلى صباح الغد.'),
 ('de', 'Ich kann Sie mit einem Spezialisten verbinden, oder wir verschieben die Lieferung auf morgen früh.')]
