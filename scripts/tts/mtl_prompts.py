"""Multilingual prompts for Chatterbox Multilingual (V3): (language, text).

PROMPTS: three voice-agent style sentences for each of en, hi, zh, ja, es, fr, ar, de (the gate
languages), interleaved so any prefix mixes languages. MILESTONE: the reference demo's sentence in
all 23 languages (ResembleAI/Chatterbox-Multilingual-TTS-V3 app.py).
"""

GATE = ["en", "hi", "zh", "ja", "es", "fr", "ar", "de"]

_BY_LANG = {
    "en": ["Thanks for calling, how can I help you today?",
           "Your appointment is confirmed for Tuesday at three thirty in the afternoon.",
           "I can transfer you to a specialist, or we can reschedule the delivery for tomorrow morning."],
    "hi": ["कॉल करने के लिए धन्यवाद, आज मैं आपकी क्या मदद कर सकता हूँ?",
           "आपकी अपॉइंटमेंट मंगलवार दोपहर साढ़े तीन बजे के लिए पक्की हो गई है।",
           "मैं आपको किसी विशेषज्ञ से जोड़ सकता हूँ, या हम डिलीवरी कल सुबह के लिए तय कर सकते हैं।"],
    "zh": ["感谢您的来电，今天有什么可以帮您的吗？",
           "您的预约已确认，时间是星期二下午三点半。",
           "我可以帮您转接专员，或者我们可以把送货改到明天上午。"],
    "ja": ["お電話ありがとうございます。本日はどのようなご用件でしょうか？",
           "ご予約は火曜日の午後三時半に確定しました。",
           "担当者におつなぎするか、配達を明日の午前中に変更することができます。"],
    "es": ["Gracias por llamar, ¿en qué puedo ayudarle hoy?",
           "Su cita está confirmada para el martes a las tres y media de la tarde.",
           "Puedo transferirle con un especialista, o podemos reprogramar la entrega para mañana por la mañana."],
    "fr": ["Merci de votre appel, comment puis-je vous aider aujourd'hui ?",
           "Votre rendez-vous est confirmé pour mardi à quinze heures trente.",
           "Je peux vous transférer à un spécialiste, ou nous pouvons reporter la livraison à demain matin."],
    "ar": ["شكرا لاتصالك، كيف يمكنني مساعدتك اليوم؟",
           "تم تأكيد موعدك يوم الثلاثاء الساعة الثالثة والنصف بعد الظهر.",
           "يمكنني تحويلك إلى أحد المختصين، أو يمكننا تأجيل التوصيل إلى صباح الغد."],
    "de": ["Danke für Ihren Anruf, wie kann ich Ihnen heute helfen?",
           "Ihr Termin ist für Dienstag um halb vier am Nachmittag bestätigt.",
           "Ich kann Sie mit einem Spezialisten verbinden, oder wir verschieben die Lieferung auf morgen früh."],
}

PROMPTS = [(lang, _BY_LANG[lang][k]) for k in range(3) for lang in GATE]

MILESTONE = {
    "ar": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب.",
    "da": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal.",
    "de": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal.",
    "el": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube.",
    "en": "Last month, we reached a new milestone with two billion views on our YouTube channel.",
    "es": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube.",
    "fi": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme.",
    "fr": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube.",
    "he": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו.",
    "hi": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।",
    "it": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube.",
    "ja": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。",
    "ko": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다.",
    "ms": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami.",
    "nl": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal.",
    "no": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår.",
    "pl": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube.",
    "pt": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube.",
    "ru": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале.",
    "sv": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal.",
    "sw": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube.",
    "tr": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık.",
    "zh": "上个月，我们达到了一个新的里程碑。 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。",
}
