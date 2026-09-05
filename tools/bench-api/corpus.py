"""Deterministic realistic-prompt generator for bench.py.

All text here is original prose written for this harness. Prompts are composed
from a paragraph bank plus per-document questions so that RAG / summarize /
chat_long requests are answerable from the supplied context and the model's
answers can be sanity-checked by eye.

Sizing uses a word-based token estimate (~1.3 tokens/word for English prose
with the Gemma tokenizer) so generated prompts are identical whether or not
the `tokenizers` package is installed.
"""
import random

WORKLOADS = ("chat_short", "chat_long", "rag_4k", "code", "summarize", "mixed")

# Target prompt-token ranges (message content only; the server adds template tokens).
RANGES = {
    "chat_short": (30, 80),
    "chat_long": (300, 600),
    "rag_4k": (2000, 3500),
    "code": (200, 500),
    "summarize": (800, 1500),
}

TOK_PER_WORD = 1.2


def est_tokens(text: str) -> int:
    return int(len(text.split()) * TOK_PER_WORD)


# --------------------------------------------------------------------------
# Paragraph bank: (title, [paragraphs], [questions answerable from the text])
# Questions are roughly ordered like the paragraphs they refer to.
# --------------------------------------------------------------------------
DOCS = [
    (
        "Lighthouses and Their Keepers",
        [
            "Before radio beacons and satellite positioning, the lighthouse was the only reliable warning a sailor had of a dangerous coast at night. The earliest towers burned open fires of wood or coal on a raised platform, and their light was feeble and easily lost in fog. The introduction of oil lamps with polished reflectors in the eighteenth century roughly doubled the useful range, and the invention of the stepped glass lens by Augustin Fresnel in 1822 changed the craft entirely. A Fresnel lens gathers most of the light from a single flame into a horizontal beam that can be seen twenty miles out to sea.",
            "Each lighthouse was given a distinct character so that a navigator could tell one from another. Some showed a fixed white light, others flashed at intervals of five, ten, or fifteen seconds, and a few alternated white and red. The pattern was printed in the official light lists that every ship carried. A keeper who let the clockwork slow so that the flashes drifted out of time was committing a serious fault, because a captain might mistake one headland for another and turn toward the rocks instead of away from them.",
            "The keeper's day was organised around the lamp. In the evening the wicks were trimmed, the reservoir filled, and the lens polished with a soft cloth to remove soot. Through the night the keeper climbed the tower every few hours to wind the rotation weights and check the flame. At dawn the lamp was extinguished, the curtains were drawn across the lantern room to protect the lens from the sun, and the brass was cleaned. Log books recorded weather, passing vessels, and any failure of the apparatus, and these logs were inspected without warning.",
            "Rock stations, built on isolated reefs with no land around them, were the hardest postings. Supplies came by boat when the weather allowed, which in winter might mean a gap of several weeks. Two or three keepers lived together in a tower barely wider than a stairwell, taking turns at the watch. Fresh water was collected from the roof, and food was salted or tinned. Keepers on rock stations were rotated ashore more often than those on land, and their pay carried a hardship allowance.",
            "Fog was the lighthouse's great enemy, because no lens can push light through dense mist. Stations near shipping lanes were therefore fitted with fog signals: first bells and cannon, later steam whistles and compressed-air horns that gave a low note audible for several miles. The keeper had to judge when visibility had dropped enough to start the signal, and the coal-fired boilers that powered the horns needed constant tending. A busy fog season could consume more fuel than the lamp itself used in a year.",
            "Automation arrived slowly. Acetylene burners with sun valves, which turned the gas on at dusk and off at dawn without a human hand, were fitted to minor lights from the 1910s. Electric lamps followed, then remote monitoring by telephone line. By the end of the twentieth century almost every lighthouse in Europe and North America had been de-staffed. The towers remain, and many are maintained as historic monuments, but the profession of keeper, with its logs, its brass, and its long solitary watches, has effectively vanished.",
            "The romance attached to lighthouses obscures how dangerous the work could be. Keepers were lost climbing exterior ladders in gales, poisoned by fumes from faulty lamps, and drowned during relief landings when a boat was thrown against the rock. Families on shore stations lived with the tower's isolation as well; children were often boarded in the nearest town for school. Yet applications for keeper posts always exceeded vacancies, and many keepers served thirty or forty years, moving from station to station as seniority allowed.",
        ],
        [
            "According to the passage, what did Fresnel's lens change about lighthouse illumination, and roughly how far could such a light be seen?",
            "Why was it a serious fault for a keeper to let the rotation clockwork slow down?",
            "Describe the keeper's routine at dawn as given in the text.",
            "Why were rock stations considered the hardest postings?",
            "How did lighthouses deal with fog, and why was this expensive?",
        ],
    ),
    (
        "The Honeybee Colony Through the Year",
        [
            "A honeybee colony is best understood as a single organism that happens to be made of tens of thousands of bodies. In midsummer a strong hive holds a queen, perhaps fifty thousand female workers, and a few hundred male drones. The queen's only tasks are laying eggs, up to two thousand a day at the peak, and producing the pheromones that hold the colony together. Workers do everything else: cleaning cells, feeding larvae, building comb, guarding the entrance, and finally, in the last weeks of their short lives, flying out to forage.",
            "The colony's year begins in late winter, when the cluster of bees that has been huddling around the queen to keep her warm senses the lengthening days. The queen resumes laying, at first only a small patch of brood in the centre of the cluster. The bees raise the temperature of the brood nest to a steady thirty-five degrees by shivering their flight muscles, burning through the honey stores at an increasing rate. Late winter is the most common time for a colony to starve, because the stores run low just as the demand for heat is highest.",
            "Spring brings the first nectar flows from willow, dandelion, and fruit blossom, and the colony expands rapidly. Foragers returning with nectar pass it to house bees, who spread it in thin films across the comb so that the water content falls from around eighty per cent to below twenty. Only then is it capped with wax and can properly be called honey. Pollen, the colony's protein source, is packed into separate cells and lightly fermented, which preserves it and makes it digestible for the larvae.",
            "When the hive becomes crowded, the workers begin to raise new queens in special peanut-shaped cells that hang from the comb. A few days before the first new queen emerges, the old queen leaves with roughly half the workers in a swarm. The swarm gathers on a branch while scout bees search for a cavity of the right size, dry and with a small defensible entrance. Scouts advertise candidate sites with dances on the surface of the swarm, and the cluster moves only when the scouts have reached a clear consensus.",
            "Summer is the season of surplus. On a warm day with a good flow a strong colony can bring in several kilograms of nectar, and beekeepers add empty boxes above the brood nest to give the bees room to store it. Drones are tolerated during this period because a swarm's new queen must mate, and she does so in flight with a dozen or more drones from other colonies. The drones die in the act. Toward the end of summer, as the nectar flow slows, the workers drive the remaining drones out of the hive to die.",
            "Autumn preparations are thorough. The workers seal cracks with propolis, a resin collected from tree buds, reduce the size of the entrance, and evict any drones that have lingered. Brood rearing slows and finally stops, and the last generation of workers, raised on abundant pollen, is physiologically different from the summer bees: they have larger fat bodies and will live for months instead of weeks. These winter bees form the cluster that will carry the queen through to the next spring.",
            "A beekeeper's interventions follow this calendar. In spring the hive is inspected for disease and for signs of swarming preparation; in summer honey is harvested and space is added; in autumn the stores are weighed and topped up with syrup if the bees are light. Winter is a time of leaving well alone, because opening a hive in the cold breaks the cluster and can chill the brood. Experienced keepers say that the best thing to do with bees in winter is to listen at the side of the box and hear the soft roar of the cluster inside.",
        ],
        [
            "According to the passage, what are the queen's only tasks, and who does the rest of the work in the hive?",
            "Why is late winter the most common time for a colony to starve?",
            "Explain how nectar becomes honey, based on the text.",
            "How does a swarm decide where to move, as described here?",
            "What distinguishes winter bees from summer bees?",
        ],
    ),
    (
        "Bread: Fermentation and Baking",
        [
            "Bread is one of the oldest prepared foods, and its essential chemistry has not changed in several thousand years. Flour, water, salt, and a leavening agent are mixed into a dough; the dough is left to ferment until it has roughly doubled in volume; and it is then baked in a hot oven. Everything else, from the choice of grain to the shape of the loaf, is a variation on this sequence. The leavening agent is almost always yeast, whether added as a purchased culture or cultivated in a sourdough starter, and its work is what turns a dense paste into an open, airy crumb.",
            "Wheat flour is unusual among cereal flours because two of its proteins, glutenin and gliadin, combine with water to form gluten, an elastic network capable of trapping gas. Kneading aligns and strengthens this network. A well-developed dough can be stretched thin enough to see light through it without tearing, a test bakers call the windowpane. Rye and barley contain little gluten, which is why loaves made from them are denser, and why rye bread traditionally relies on the gummy pentosan sugars in the grain rather than on gluten to hold its structure.",
            "Yeast consumes the simple sugars in the dough and produces carbon dioxide and alcohol. The carbon dioxide inflates the bubbles that kneading has created; the alcohol evaporates in the oven, contributing to the aroma of fresh bread. Fermentation is slower in cold conditions and faster in warm ones, and bakers manipulate temperature deliberately. A dough left overnight in a cold room develops far more flavour than one raised quickly in a warm kitchen, because bacteria that live alongside the yeast have time to produce lactic and acetic acids.",
            "Sourdough is bread leavened by a starter, a stable culture of wild yeast and lactic acid bacteria maintained by regular feeding with flour and water. The bacteria acidify the dough, which gives sourdough its characteristic tang, slows the growth of spoilage organisms, and alters the gluten so that the finished loaf keeps for days without going stale. A starter that is fed on a strict schedule and kept at a consistent temperature will behave predictably for years. Bakers often keep the same starter for decades, passing it between shops and households.",
            "Shaping a loaf serves a structural purpose. When the fermented dough is folded and rolled into a tight ball or cylinder, the outer surface is stretched into a skin that holds the shape during the final rise and directs the expansion in the oven. Loaves are usually slashed with a blade just before baking to control where this expansion breaks the crust. Without a slash, the loaf bursts wherever the skin is weakest, often along the side, and the crumb inside is compressed.",
            "In the oven the loaf first expands rapidly as the gas in the bubbles heats and the last burst of yeast activity releases more carbon dioxide; bakers call this oven spring. At around sixty degrees the starches in the crumb gelatinise and the structure sets. The crust browns through the Maillard reaction between sugars and amino acids, and finally through caramelisation. Steam in the first minutes of baking keeps the crust soft long enough for the loaf to reach its full volume, which is why professional ovens inject steam and home bakers use covered pots.",
            "Bread stales not because it dries out but because the gelatinised starch slowly recrystallises, a process called retrogradation that firms the crumb within a day or two. Refrigeration accelerates this, which is why bread should never be stored in the fridge. Freezing halts it. A stale loaf can be partially revived by heating it, because warmth reverses the crystallisation temporarily, but the effect does not last once the loaf cools again.",
        ],
        [
            "Why does wheat flour produce a lighter loaf than rye or barley, according to the passage?",
            "What does the passage say about the effect of fermenting dough overnight in a cold room?",
            "What role does the starter's acidity play in sourdough?",
            "Why are loaves slashed before baking?",
            "According to the text, why should bread not be stored in a refrigerator?",
        ],
    ),
    (
        "Glaciers and the Shaping of Valleys",
        [
            "A glacier forms wherever more snow falls in winter than melts in summer, year after year, until the accumulated layers compress into ice. Fresh snow is mostly air; as it is buried its crystals recrystallise into granular firn, and at a depth of several tens of metres the air pockets close off and the firn becomes solid glacier ice. Under its own weight this ice deforms and flows downhill like an extremely viscous fluid. Even a small mountain glacier may move several tens of metres in a year, and the great ice streams of Greenland move kilometres.",
            "The ice moves in two ways. Within the body of the glacier, individual crystals slip along their internal planes, so that the ice creeps. At the base, where the pressure and friction produce a thin film of meltwater, the whole mass may slide over the bedrock. Glaciers whose beds are at the melting point slide much faster than glaciers frozen to their beds, which is one reason why temperate glaciers in the Alps or New Zealand are more effective at eroding valleys than the cold-based glaciers of Antarctica.",
            "Erosion happens by two mechanisms. Rock fragments frozen into the base of the ice are dragged across the bedrock, scratching and polishing it; the parallel grooves left behind, called striations, record the direction of flow. Meanwhile meltwater seeping into cracks in the bedrock freezes and expands, loosening blocks that the moving ice then plucks away. Plucking is most effective on the downstream side of obstacles, so glaciated bedrock often shows a smooth, gently sloping upstream face and a steep, jagged downstream face.",
            "The most recognisable result of glacial erosion is the U-shaped valley. A river cuts a V-shaped notch, because its erosion is concentrated in the narrow channel at the bottom. A glacier fills the valley from wall to wall and erodes across the whole width, deepening and straightening the trough and leaving steep sides and a broad flat floor. Tributary valleys that carried smaller glaciers were not deepened as much, so when the ice melts they are left hanging high above the main valley floor, and their streams descend as waterfalls.",
            "At the head of a glacier, where snow accumulates in a hollow on the mountainside, erosion produces an armchair-shaped basin called a cirque. When cirques form on several sides of a peak and grow back toward one another, they leave a sharp pyramidal summit, of which the Matterhorn is the textbook example. Where two adjacent cirques meet they form a narrow knife-edged ridge. These features can be recognised in mountain ranges that have been free of ice for ten thousand years.",
            "Everything the glacier erodes is eventually deposited. At the glacier's snout, where the ice melts, the debris it has carried is dumped in a ridge called a terminal moraine, which may dam the valley and impound a lake. Along the sides, debris that fell from the valley walls onto the ice accumulates as lateral moraines. Beneath the ice, meltwater streams sort the sediment into ridges of sand and gravel called eskers. The unsorted mixture of clay, sand, and boulders that the ice itself leaves behind is called till, and much of the farmland of northern Europe and North America lies on it.",
            "Reading a glaciated landscape is a matter of matching these features to the flow. Striations and the asymmetry of plucked outcrops give the direction; the height of lateral moraines on the valley walls gives the former thickness of the ice; and the position of the terminal moraine gives its furthest extent. From such evidence geologists have reconstructed the ice sheets of the last glacial maximum, which twenty thousand years ago covered most of Canada and Scandinavia under ice more than two kilometres thick.",
        ],
        [
            "Explain the two ways in which a glacier moves, according to the passage.",
            "What are striations and what do they record?",
            "Why do glaciers leave U-shaped valleys while rivers leave V-shaped ones?",
            "What is a hanging valley and how does it form?",
            "What is the difference between till and an esker, based on the text?",
        ],
    ),
    (
        "The Printing Press and the Spread of Books",
        [
            "Before the middle of the fifteenth century, every book in Europe was written by hand. A large volume such as a complete Bible took a trained scribe most of a year, and the parchment for it required the skins of a couple of hundred animals. Books were therefore rare, expensive, and concentrated in monasteries, universities, and the households of the very wealthy. A scholar might travel for weeks to consult a single manuscript, and errors introduced by one copyist were faithfully reproduced by the next.",
            "The technology that changed this was not a single invention but a combination. Paper, made from rags, had arrived from the Islamic world and was far cheaper than parchment. The wooden screw press had long been used for wine and olives. Metal casting was well understood by goldsmiths. What Johannes Gutenberg of Mainz contributed, around 1450, was a way of casting large numbers of identical metal letters from an adjustable mould, an oil-based ink that would stick to metal, and the practical organisation of the whole into a workshop that could produce a page many times faster than a scribe.",
            "Typesetting was slow but it needed to be done only once per page. A compositor picked individual letters from a case and arranged them, in reverse, in a frame called a forme. The forme was locked into the press, inked with leather pads, and a sheet of dampened paper was laid over it. Pulling the lever pressed the paper onto the type. Two workers at a press could produce perhaps two hundred and fifty impressions an hour, and a book of two hundred pages in an edition of five hundred copies could be finished in a few months.",
            "The economics were transformative. The cost of a printed book was a fraction of a manuscript, and it fell further as editions grew. Within fifty years of Gutenberg's Bible, presses were operating in more than two hundred European towns, and several million volumes had been printed. Latin classics, law books, and religious texts came first, because their market was assured, but printers soon discovered that works in the vernacular languages sold to a far larger public.",
            "Printing also standardised. Because every copy of an edition was identical, a scholar in Krakow and a scholar in Lisbon could refer to the same page of the same text and be sure they were reading the same words. Errors still occurred, but they could be corrected in the next edition rather than propagated indefinitely. Maps, anatomical diagrams, and mathematical tables, which are almost impossible to copy accurately by hand, could now be reproduced exactly, and this made cumulative technical progress possible in a way it had not been before.",
            "The press changed the shape of the book as an object. Manuscripts had no title pages and rarely a table of contents; printed books acquired both, along with page numbers, running heads, and indexes, because a large edition needed a standard way to be identified and navigated. Typefaces modelled on the humanist handwriting of Italian scribes displaced the dense gothic letter, and the roman and italic faces still in use today descend directly from those cut in Venice in the 1470s and 1490s.",
            "Authorities quickly realised that the press could spread ideas as fast as it spread texts. Licensing systems, lists of forbidden books, and the requirement that printers register their names on every title page appeared within decades. These measures slowed but did not stop the traffic in pamphlets, which were cheap, quickly printed, and easily concealed. The religious controversies of the sixteenth century were conducted largely in print, and the modern notion of public opinion, formed by readers who never meet, is a child of the printing press.",
        ],
        [
            "According to the passage, what did Gutenberg actually contribute, given that paper, presses, and metal casting already existed?",
            "Describe the steps of printing a page as given in the text.",
            "Why did printing make cumulative technical progress possible, according to the passage?",
            "What features of the modern book does the text say were introduced because of printing?",
            "How did authorities respond to the press, and how effective were their measures?",
        ],
    ),
    (
        "Down the River: A Journey by Small Boat",
        [
            "We put the boat in at the old ferry landing just after sunrise, when the water was still flat and the mist had not yet lifted off the reeds. There were three of us and enough food for a week, though the river towns were never more than a day apart and we could have resupplied anywhere. The current at the landing was barely perceptible. It was only when I looked back at the shore and saw the ferry slip sliding away that I realised we were already moving.",
            "The upper river is shallow and braided, splitting around gravel bars and rejoining in a way that the chart only approximately describes. Twice on the first morning we ran aground and had to climb out and walk the boat into deeper water, the gravel shifting under our boots. Herons stood at the edge of every bar and lifted off, unhurried, as we approached. By noon the channels had joined into one, the banks had risen into low clay bluffs, and the river had begun to feel like a river.",
            "The first town announced itself by a church tower and a smell of woodsmoke. We tied up below a stone quay where two men were mending a net and asked about the weir we had been told lay downstream. One of them drew it for us on the back of a receipt: keep to the left bank, take the old lock cut, and do not on any account go over the sill, which had wrecked a canoe the summer before. The lock keeper, he said, would be at lunch until two, and we would do well to wait for him rather than work the gates ourselves.",
            "Below the weir the river changed character again. It ran between wooded hills, deep and dark green, and the only sounds were our paddles and the occasional splash of a fish. We passed a heronry, a dozen untidy nests high in a dead elm, and a place where the bank had collapsed and taken a stretch of fence into the water. Toward evening a kingfisher flew ahead of us for half a mile, perching, waiting, and darting on again, as if it had been assigned to show us the way.",
            "We camped that night on a shingle beach on the inside of a bend, where the driftwood was plentiful and the ground was flat. The river talked all night, a steady conversation of small sounds that seemed to change whenever I stopped listening. In the morning there was frost on the tent and the water was steaming. The stove took three matches to light. We were on the water again before the sun had cleared the hills, our breath hanging in the air ahead of us.",
            "The middle river is farming country, and the banks were lined with willows planted long ago to hold the soil. Cattle came down to drink and watched us pass with blank attention. At a place where a tributary joined from the north the water became visibly browner and faster, and for an hour we had only to steer. We saw our first barge here, a long low vessel carrying gravel, whose wash lifted us and set us down again as it went by. Its skipper raised one hand without looking at us.",
            "By the fourth day the river was wide enough that the far bank was a line rather than a place, and the tide had begun to reach us. Twice a day the current slackened, stopped, and turned, and we learned to time our departures to the ebb. The towns were larger, with warehouses and cranes, and the water smelled faintly of salt. On the last morning we rounded a point and saw, low on the horizon, a grey line that was not cloud. It was the sea, and the journey was over, though the river went on without us.",
        ],
        [
            "What difficulties did the travellers meet on the upper river, according to the passage?",
            "What advice did the man mending nets give about the weir?",
            "How did the river change below the weir?",
            "How did the tide affect the travellers' journey on the lower river?",
            "How did the travellers know they had reached the end of their journey?",
        ],
    ),
    (
        "Tides and the Moon",
        [
            "The tides are the most visible daily reminder that the Earth is not alone in space. Twice in roughly twenty-five hours the sea rises and falls along every coast, sometimes by a few centimetres and sometimes, in narrow funnel-shaped bays, by more than fifteen metres. The cause is the gravitational pull of the Moon and, to a lesser extent, the Sun, acting on the oceans. The Moon's pull is stronger on the side of the Earth facing it than on the far side, and this difference stretches the ocean into two bulges, one toward the Moon and one away from it.",
            "The Earth rotates once a day beneath these two bulges, so a given point on the coast passes through high water twice. Because the Moon also moves along its orbit in the same direction as the Earth's rotation, the Earth must turn a little further each day to catch up, and the tides arrive about fifty minutes later than they did the day before. Anyone who has spent a week at the seaside will have noticed the low tide creeping from morning toward afternoon over the course of their stay.",
            "The Sun raises tides in the same way, but its effect is less than half the Moon's, because although the Sun is enormously more massive, it is also very much further away, and tidal force falls off with the cube of distance. When Sun and Moon are aligned, at new moon and full moon, their tides add and the range is greatest; these are spring tides, a name that has nothing to do with the season. When they are at right angles, at the quarter moons, their tides partly cancel and the range is smallest; these are neap tides.",
            "If the Earth were covered by a uniform ocean, the tides would be simple. In reality the continents block the bulges, and the water in each ocean basin sloshes in a pattern determined by the basin's shape and depth. The result is that in some places, such as parts of the Gulf of Mexico, there is only one high tide a day, while in others the two daily highs are of very unequal height. Tidal predictions for a given port are therefore made not from first principles but from decades of observations, analysed into a set of component waves whose sum is projected forward.",
            "In shallow, narrowing bays the tide can be amplified enormously. The incoming water is squeezed into a smaller and smaller cross-section, and if the length of the bay happens to match the natural period of the tide, the water resonates like air in an organ pipe. The Bay of Fundy in Canada and the Severn estuary in Britain are the famous examples, with ranges above twelve metres. In the Severn the rising tide can outrun itself and form a wave, the bore, which travels many kilometres upriver and is ridden by surfers.",
            "Tides matter to more than sailors. Intertidal life is organised in horizontal bands according to how long each species can survive out of water; barnacles and periwinkles high on the rocks may be exposed for hours, while kelp at the bottom of the shore is uncovered only at the lowest spring tides. Estuaries flush their pollution on the ebb and receive nutrients on the flood. Tidal power stations, which trap the high water behind a barrage and release it through turbines, exploit the same predictability that made tide tables possible.",
            "The tides also have a slow, one-directional effect. Friction between the moving water and the sea floor dissipates energy, and the drag of the tidal bulges on the rotating Earth is gradually lengthening the day by about two milliseconds per century. The same interaction pushes the Moon outward, so that it recedes from the Earth by nearly four centimetres a year. Fossil corals, whose growth bands record days and years, confirm that four hundred million years ago the day was about twenty-two hours long.",
        ],
        [
            "Why are there two high tides a day rather than one, according to the passage?",
            "Why do the tides arrive about fifty minutes later each day?",
            "What is the difference between spring tides and neap tides, and what causes it?",
            "Why can't tide tables be computed from first principles, according to the text?",
            "How are the tides slowly changing the length of the day and the distance to the Moon?",
        ],
    ),
    (
        "Steam Engines and the Early Railways",
        [
            "The first steam engines were not built to move anything. They were pumps, installed at the heads of mine shafts in the early eighteenth century to lift water out of workings that had gone below the level natural drainage could reach. Thomas Newcomen's engine of 1712 admitted steam to a cylinder, condensed it with a spray of cold water so that the atmospheric pressure outside pushed the piston down, and used a rocking beam to work a pump rod in the shaft. It was enormous, slow, and wasteful of coal, but coal at a colliery was almost free.",
            "James Watt's improvement, patented in 1769, was to condense the steam in a separate vessel so that the working cylinder could stay hot. This roughly quartered the fuel consumption and made steam power economical away from the coalfields, in textile mills and breweries. Watt also devised the mechanism that converted the beam's rocking motion into rotation, so that an engine could drive machinery through a shaft. By 1800 there were several hundred Watt engines at work, and the cotton industry had begun its migration from waterside villages to the towns.",
            "Watt's engines ran at low pressure, barely above that of the atmosphere, because he distrusted the boilers of his day. The next generation of engineers, led by Richard Trevithick in Cornwall, accepted the risk of higher pressure and gained an engine small and powerful enough to move itself. Trevithick's locomotive of 1804 hauled ten tons of iron along a tramway in South Wales. It was a demonstration rather than a commercial success, because the cast-iron rails broke under its weight, but it settled the question of whether smooth wheels could grip smooth rails.",
            "The commercial railway was born at the collieries of north-east England, where horse-drawn wagonways had carried coal to the rivers for two centuries. George Stephenson, an engine-wright at one of these pits, built locomotives that were reliable enough to replace the horses, and in 1825 the Stockton and Darlington line opened with his engine hauling both coal and, unexpectedly, passengers. The multi-tube boiler, in which hot gases from the fire pass through many small tubes surrounded by water, was the decisive advance; it produced steam fast enough for sustained speed.",
            "The Liverpool and Manchester Railway of 1830 was the first line planned from the start for locomotives and for passengers, and it was an immediate success. Journeys that had taken a day by canal barge took two hours. Within twenty years Britain had more than six thousand miles of track, and the mania for building lines spread to the continent and to North America. The scale of construction was unprecedented: cuttings, embankments, tunnels, and viaducts were built by hand by gangs of labourers who moved from project to project and lived in shanty camps beside the works.",
            "Railways demanded standardisation. Trains running between towns needed a common gauge for their rails, and after a long dispute the British parliament fixed it in 1846 at four feet eight and a half inches, the width Stephenson had inherited from the colliery wagonways. Timetables were impossible while every town kept its own local time, so the companies adopted a single railway time based on Greenwich, and the towns followed. The telegraph, strung along the track to signal between stations, became the first long-distance communication network.",
            "The social effects were as large as the economic ones. Fresh milk and fish reached inland cities. Newspapers printed in London were read the same day in Manchester. Seaside towns grew up to receive excursion trains full of factory workers. People who had never travelled more than a few miles from their birthplace could visit relatives across the country. Critics warned of the dangers of speed and the loss of local character, but by the middle of the century the railway had become the ordinary background of life rather than a marvel.",
        ],
        [
            "What was the original purpose of the first steam engines, according to the passage?",
            "What was Watt's key improvement and what effect did it have on fuel consumption?",
            "Why was Trevithick's 1804 locomotive not a commercial success?",
            "What was the decisive technical advance that made sustained locomotive speed possible?",
            "Why did the railways lead to the adoption of a single standard time?",
        ],
    ),
    (
        "Household Accounts: A Primer on Budgeting",
        [
            "A household budget is nothing more than a plan for money that has not yet arrived. Its purpose is not to restrict spending for its own sake but to make sure that the things that matter most are paid for first and that surprises can be absorbed without borrowing. The oldest and simplest method is to divide income into a small number of envelopes at the start of each month, one for rent, one for food, one for fuel and light, and so on, and to spend from each envelope only what it contains. Modern budgeting software does the same thing with categories instead of envelopes.",
            "The first step is to know what actually comes in and goes out, and this is harder than it sounds. Income may be irregular, and small daily expenses are easily forgotten. The standard advice is to record every transaction for at least a month without trying to change anything, and then to sort the results into fixed costs, which are the same every month and hard to alter quickly, and variable costs, which respond to decisions. Most people are surprised by how much falls into a third category, occasional costs such as repairs, gifts, and annual fees, which are predictable in total even though each one looks like a surprise.",
            "Fixed costs deserve the most scrutiny precisely because they are the hardest to change. Rent or a mortgage, insurance, subscriptions, and loan repayments recur whether or not the month has gone well, and a household whose fixed costs consume most of its income has no room to manoeuvre. A common rule of thumb is that fixed costs should be no more than half of take-home pay, with the remainder split between variable spending and saving. The rule is arbitrary, but it captures the idea that flexibility itself has value.",
            "Occasional costs are best handled by turning them into fixed ones. If a household spends around six hundred in a year on car maintenance, setting aside fifty each month into a separate fund means that the bill, when it arrives, is already paid. The same applies to holidays, clothing, and the replacement of appliances. This is sometimes called a sinking fund, a term borrowed from public finance, where it originally described money set aside each year to pay off a debt at maturity.",
            "An emergency fund is a sinking fund for the unknown. Its purpose is to cover a period without income or an unavoidable large expense without resort to credit. Three months of essential spending is the figure most often recommended, though households with irregular income or dependants may want more. The fund should be kept somewhere accessible but not too accessible; a separate savings account that takes a day to transfer from is ideal, because it removes the temptation to treat it as ordinary money.",
            "Debt changes the arithmetic. Interest on borrowing is a fixed cost that produces nothing, and the higher the rate, the more urgent it is to eliminate. Two strategies are commonly advised. The first pays off the highest-interest debt first, which minimises the total interest paid. The second pays off the smallest debt first regardless of rate, which minimises the number of creditors quickly and is easier to sustain. The mathematical answer is the first; the behavioural evidence often favours the second.",
            "A budget that is never reviewed is a wish rather than a plan. A short monthly reconciliation, comparing what was planned with what happened, is the habit that makes the whole thing work. Categories that are always overspent should be enlarged and the money found elsewhere; categories that are never spent can be reduced. Over a year the budget comes to describe the household as it actually is, and at that point it stops feeling like a constraint and begins to feel like a description.",
        ],
        [
            "What are the three categories of costs described in the passage, and why is the third one surprising?",
            "Why does the text say fixed costs deserve the most scrutiny?",
            "What is a sinking fund and how does it apply to household budgeting?",
            "Where does the passage recommend keeping an emergency fund, and why?",
            "What two debt repayment strategies are described, and which does the text say the evidence favours?",
        ],
    ),
    (
        "Reading the Sky: Clouds and Local Forecasting",
        [
            "Long before there were weather services, farmers, sailors, and shepherds forecast the weather by watching the sky, and their methods were sound because clouds are the visible signature of what the air is doing. Air that rises cools, and if it cools below its dew point the water vapour in it condenses into droplets. Every cloud therefore marks a place where air is going up. The shape of the cloud tells you how it is rising: gently and over a wide area, or violently and in a narrow column.",
            "The high clouds are made of ice. Cirrus, the thin wisps and hooks sometimes called mares' tails, form at eight or ten kilometres and are often the first sign of an approaching warm front, arriving a day or more ahead of the rain. When cirrus thickens into a milky sheet through which the sun shows a halo, the front is closer. The old saying that a ring around the sun or moon means rain within a day is a fair description of a warm front advancing over a station.",
            "The middle clouds, altostratus and altocumulus, occupy the layer between two and seven kilometres. Altostratus is a grey featureless sheet through which the sun appears as if through frosted glass; it usually means that rain is a few hours away. Altocumulus appears as rounded masses or rolls, sometimes arranged in lines, and when these masses grow into little turrets on a warm morning they indicate an unstable atmosphere and a fair chance of thunderstorms by afternoon.",
            "The low clouds are stratus, a uniform grey layer that produces drizzle or nothing, and stratocumulus, a lumpy layer with gaps through which blue sky may show. Stratus that forms at ground level is fog. Neither brings much weather of its own; they are what the sky looks like under a stable air mass, and the forecast they imply is more of the same. In winter a persistent stratocumulus deck can keep a region dull and cold for a week while the sun shines a few hundred metres above it.",
            "Cumulus clouds are the product of convection, columns of warm air rising from ground heated by the sun. Small fair-weather cumulus, with flat bases and cauliflower tops, appear on most warm afternoons and dissolve toward evening. If the air above is unstable the columns keep rising and the clouds grow into towering cumulus and finally cumulonimbus, the thunderhead, whose flat anvil top marks the point where the rising air has hit the stable stratosphere and spread out sideways. A cumulonimbus brings heavy rain, hail, gusty winds, and lightning, and can develop in an hour.",
            "Wind direction adds a second line of evidence. In the northern hemisphere, air circulates anticlockwise around a low-pressure centre, so if you stand with your back to the wind the low is on your left. A wind that backs, shifting anticlockwise from west toward south, generally means a low is approaching and weather will deteriorate; a wind that veers, shifting clockwise, means it is passing and conditions will improve. Sailors combined this rule with the barometer, whose fall or rise measured how fast the change was coming.",
            "None of these signs is infallible, and the skill of local forecasting lies in combining them and in knowing the peculiarities of the place. Coasts have sea breezes that pull cumulus inland in the afternoon; valleys pool cold air at night and fill with fog; mountains force air upward and wring rain from clouds that would have passed harmlessly over the plain. An observer who has watched one sky for a few years will often outperform a general forecast for that particular spot, simply by knowing what the sky there has done before.",
        ],
        [
            "According to the passage, what does every cloud indicate about the air, and what does its shape tell you?",
            "What does a halo around the sun or moon signify, and why?",
            "How does a cumulonimbus cloud form and what does its anvil top indicate?",
            "Explain the rule about wind backing and veering given in the text.",
            "Why can a local observer sometimes outperform a general forecast?",
        ],
    ),
    (
        "Coffee: From Bean to Cup",
        [
            "The coffee plant is a small evergreen tree of the highlands of East Africa, cultivated today across a belt of tropical countries between the two tropics. It produces white, jasmine-scented flowers and then fruits about the size of a cherry that ripen from green through yellow to deep red over several months. Each cherry normally contains two seeds, flat sides together, and these seeds, once fermented, dried, and roasted, are the coffee beans of commerce. Two species dominate: arabica, grown at altitude and prized for flavour, and robusta, hardier and higher in caffeine.",
            "Harvesting is done largely by hand, because the cherries on a single branch ripen at different times and machines cannot tell them apart. Pickers pass through the same trees several times over the season, taking only the red fruit. A skilled picker may gather fifty kilograms of cherry in a day, which after processing yields about ten kilograms of green beans. On steep hillsides in Ethiopia, Colombia, or Guatemala this labour is the largest single cost of production and the reason coffee remains a smallholder crop.",
            "The fruit must be removed from the seed within hours of picking. In the washed process the cherries are pulped by machine, and the seeds, still coated in sticky mucilage, are left in tanks of water for a day or two while bacteria and yeast break the mucilage down. The seeds are then washed and dried on raised beds or patios. In the natural process the whole cherry is simply spread out to dry in the sun for several weeks, and the dried fruit is milled off afterward. Washed coffees tend to be cleaner and brighter; naturals are heavier, sweeter, and more variable.",
            "Green coffee has almost none of the aroma of the finished drink. Roasting develops it. As the beans are heated to around two hundred degrees they lose water, swell, and turn from green to yellow to brown; at a certain point the internal pressure cracks the bean with an audible snap, and the sugars and amino acids begin the reactions that produce the hundreds of aromatic compounds in roasted coffee. A light roast stops soon after this first crack and preserves the origin's acidity; a dark roast continues until the oils come to the surface and the flavour is dominated by the roast itself.",
            "Roasted coffee is perishable. Carbon dioxide trapped in the bean escapes over the first few days, carrying aromatics with it, and oxygen begins to turn the oils rancid within a couple of weeks. Grinding accelerates both processes many times over by exposing more surface. The advice to buy whole beans in small quantities and grind just before brewing is not snobbery but chemistry. Bags with one-way valves allow the gas to escape without letting air in, which is why fresh coffee is packed that way.",
            "Brewing is extraction. Hot water dissolves the soluble compounds in the ground coffee, and the taste of the cup depends on how much of what is dissolved. Too little extraction gives a thin, sour cup; too much gives a bitter, harsh one. The variables are grind size, water temperature, contact time, and the ratio of coffee to water, and every brewing method is a particular combination of them. Espresso forces water through a fine grind under pressure in under thirty seconds; a filter brew drips through a coarser grind over several minutes; cold brew steeps a very coarse grind for many hours at room temperature.",
            "Coffee's spread from the Ethiopian highlands to every corner of the world took about five hundred years. It was drunk in Yemen by the fifteenth century, in Istanbul and Cairo by the sixteenth, and in Venice, London, and Amsterdam by the seventeenth, where the coffee house became a place for news, business, and argument. The Dutch and French carried the plant to their tropical colonies, and by the nineteenth century Brazil had become, and remains, the largest producer. Today coffee is the most widely traded agricultural commodity after grain.",
        ],
        [
            "What are the two main species of coffee described in the passage, and how do they differ?",
            "Why is coffee harvested by hand, according to the text?",
            "Explain the difference between the washed and natural processes.",
            "What happens at the first crack during roasting?",
            "According to the passage, why should coffee be ground just before brewing?",
        ],
    ),
    (
        "Keeping Time: From Sundials to Pendulum Clocks",
        [
            "The oldest way to tell the time is to watch the shadow of a stick. A sundial is simply a stick, the gnomon, arranged so that its shadow falls across a marked plate, and dials of this kind were in use in Egypt and Babylon more than three thousand years ago. The difficulty is that the sun's apparent motion is not uniform through the year, so an hour marked on a plain dial in summer is longer than one in winter. Ancient hours were therefore elastic, twelve to a day and twelve to a night regardless of the season.",
            "Water clocks solved the problem of night and cloud. A vessel with a small hole in the bottom empties at a rate that depends mainly on the depth of water above the hole, and by marking the inside of the vessel one can read the time as the level falls. The Greeks and later the Arabs built elaborate versions with floats, gears, and figures that struck bells. Their accuracy was limited by the changing flow rate as the vessel emptied, and by the freezing of water in winter, which is one reason sand was substituted in the hourglass.",
            "The mechanical clock appeared in Europe in the late thirteenth century, and its heart was the escapement, a device that lets a falling weight turn a wheel not continuously but in a series of small, equal steps. The earliest escapement, the verge and foliot, used a horizontal bar with weights on its ends that swung back and forth, alternately catching and releasing the teeth of the wheel. It kept time to within perhaps a quarter of an hour a day, which was good enough to ring the bells of a monastery or town hall but useless for anything finer.",
            "The great leap came from the observation, attributed to Galileo, that a pendulum swings in almost the same time whatever the size of its swing. A pendulum is therefore a natural timekeeper, and in 1656 Christiaan Huygens built the first clock regulated by one. The improvement was startling: errors fell from a quarter of an hour a day to about a minute. Within a generation the pendulum clock had acquired a minute hand, and then a second hand, because for the first time there was something worth measuring at that scale.",
            "Pendulum clocks have two enemies. The first is temperature, because a metal rod lengthens as it warms and a longer pendulum swings more slowly; a clock gains in winter and loses in summer. The remedies were rods of wood, which expand little, or compound pendulums of brass and steel arranged so that their expansions cancel. The second is the escapement's own interference, since every push it gives the pendulum slightly disturbs its swing. The best regulators of the eighteenth century, with these problems addressed, kept time to a few seconds a week.",
            "Pendulums cannot work at sea, where the motion of the ship swamps their swing, and the problem of finding longitude drove the development of a different kind of timekeeper. Longitude is a matter of comparing local noon, found from the sun, with the time at a reference meridian, and the reference time had to be carried on board. John Harrison's marine chronometers, perfected in the 1760s, used a balance wheel and spring compensated for temperature, and kept time to within a few seconds over a voyage of months. A copy of Harrison's fourth timekeeper went with Cook to the Pacific.",
            "The pattern of these developments repeats throughout the history of measurement. Each new standard, once available, revealed problems that the old one had hidden: the pendulum exposed the irregularity of the sun's day, the chronometer exposed the variations in the pendulum, and the quartz and atomic clocks of the twentieth century exposed the slow wobble of the Earth's rotation itself. Time is now defined by the vibration of caesium atoms, and the rotation of the planet is checked against it rather than the other way round.",
        ],
        [
            "Why were ancient hours elastic, according to the passage?",
            "What limited the accuracy of water clocks?",
            "What is an escapement and what does it do?",
            "Why did the pendulum clock lead to the addition of minute and second hands?",
            "Why did the search for longitude require a different kind of timekeeper than the pendulum?",
        ],
    ),
]

# --------------------------------------------------------------------------
# chat_short
# --------------------------------------------------------------------------
SHORT_QUESTIONS = [
    "What is the difference between baking soda and baking powder, and can I substitute one for the other in a muffin recipe?",
    "My sourdough starter smells like nail polish remover after two days in the fridge. Is it dead, or can I revive it?",
    "Why does the tide come in later each day, and by roughly how much?",
    "Explain in plain language why glaciers carve U-shaped valleys while rivers carve V-shaped ones.",
    "How did lighthouse keepers stop sailors from confusing one lighthouse with another at night?",
    "What's a reasonable size for an emergency fund if my income is irregular, and where should I keep it?",
    "I have credit card debt at 22% and a car loan at 6%. Which should I pay off first and why?",
    "What does a halo around the moon usually mean for tomorrow's weather?",
    "Why do some coffee bags have a little plastic valve on them?",
    "What does the 'first crack' mean when roasting coffee beans?",
    "How did the printing press change the way books looked, not just how many there were?",
    "Why did early railways force towns to give up their local time?",
    "Why can't a pendulum clock be used to find longitude at sea?",
    "What's the simplest way to explain what an escapement does in a mechanical clock?",
    "Why do bees kick the drones out of the hive at the end of summer?",
    "How does a swarm of bees decide where to build its new home?",
    "Why shouldn't I keep bread in the fridge if I want it to stay fresh?",
    "What's the windowpane test when kneading dough, and what does it tell me?",
    "Why is a spring tide called that if it has nothing to do with the season?",
    "What is a terminal moraine and why does it sometimes create a lake?",
    "Give me three practical tips for reading tomorrow's weather from today's clouds.",
    "What are the tradeoffs between washed and natural processed coffee?",
    "In two or three sentences, what did Watt actually improve about the steam engine?",
    "Why were the first steam engines built at coal mines rather than in factories?",
    "What is propolis and what do bees use it for?",
    "How long does it take for fresh snow to become glacier ice, and what happens along the way?",
    "Why does a fog signal at a lighthouse burn so much fuel?",
    "What is the 50% rule for fixed costs in a household budget, and is it a good rule?",
    "Explain oven spring to someone who has never baked bread.",
    "What are cirrus clouds made of, and what do they usually foretell?",
    "What does it mean when the wind 'backs' versus 'veers', and which is the bad sign?",
    "Why did Trevithick's first locomotive break the rails it ran on?",
    "Why does grinding coffee make it go stale faster?",
    "Write a two-sentence explanation of why the Moon is slowly moving away from the Earth.",
    "What's the difference between till and an esker?",
    "What's a sinking fund, in a household context, and how do I set one up?",
    "Why did Fresnel's lens matter so much for lighthouses?",
    "What are the four main variables in brewing coffee that affect extraction?",
    "Why does a printed book have a title page when medieval manuscripts usually didn't?",
    "Name two ways bakers control where a loaf's crust splits in the oven.",
]

SHORT_PREFIX = [
    "Quick question: ",
    "Hi! ",
    "I'm a complete beginner at this, so please keep the explanation simple. ",
    "I'm writing a short explainer for a community newsletter and need a clear, accurate answer. ",
    "My ten-year-old asked me this at dinner and I realised I wasn't sure. ",
    "Settle a debate between me and a friend. ",
    "I've read conflicting things online about this. ",
    "I'm preparing notes for a talk at our local club next week. ",
    "Bear with me, this might be a basic question. ",
    "Context: I'm a hobbyist, not a professional. ",
]

SHORT_SUFFIX = [
    "",
    " Keep it under 100 words.",
    " Answer in two or three sentences.",
    " Use a numbered list if that helps.",
    " Please be concrete rather than general, and avoid jargon.",
    " If there's a common misconception here, mention it.",
    " A short example would help me remember it.",
    " Start with the one-sentence answer, then elaborate briefly.",
    " If it depends on circumstances, say what it depends on.",
]

# --------------------------------------------------------------------------
# chat_long scenarios (persona + situation) with follow-up questions
# --------------------------------------------------------------------------
SCENARIOS = [
    (
        "I run a small bakery that opens at seven in the morning. At the moment I mix my sourdough at four in the afternoon, bulk ferment it on the counter until about nine at night, shape it, and put it in the fridge overnight, then bake straight from cold at five in the morning. The loaves taste good but the crumb is tighter than I'd like and the crust sometimes splits along the side instead of where I've scored it. The kitchen is about 24 degrees in the afternoon and closer to 19 by evening. I use a stone-ground bread flour at around 78% hydration and a starter that I feed twice a day.\n\nA few more details in case they matter: I bake in a deck oven at 240 degrees with steam for the first twelve minutes, loaves are about 900 grams, and I do two sets of stretch-and-folds in the first hour of bulk. The starter is rye-based and usually peaks about five hours after feeding. I've tried a longer bulk once, but the dough went slack and the loaves spread. I'd rather not change flour since customers like the flavour.",
        [
            "What changes to my schedule or process would you try first to open up the crumb, and why?",
            "Why might the crust be splitting along the side, and how do I fix it?",
            "If I wanted to shorten the whole process by three hours without losing flavour, where would you cut?",
        ],
    ),
    (
        "My partner and I bring home about 5,200 a month after tax between us. Rent is 1,900, the car payment is 380, insurance of all kinds is about 300, and we have subscriptions that add up to roughly 90. We spend around 700 on groceries, 250 on fuel, and honestly probably 400 on eating out and small stuff we don't track. We have one credit card with 3,100 on it at 24% and a student loan of 11,000 at 5%. Savings are about 1,200 in a current account. We'd like to build an emergency fund and start saving for a house deposit within the next two years.\n\nSome extra context: one of us is self-employed, so income can swing by a few hundred either way month to month, and there's a tax bill due every January that we always scramble for. The car will need new tyres and a service in the next few months, probably 600 or so. We have no children and no other debts. Neither of us has ever actually kept a budget; we just check the balance before big purchases.",
        [
            "Lay out a monthly budget for us using these numbers, and tell us what you would change first and why.",
            "Should we pay down the credit card before building the emergency fund, or do both at once? Give a concrete plan.",
            "Which of our costs count as fixed, variable, and occasional, and what would you do about the untracked spending?",
        ],
    ),
    (
        "I've kept two hives for one season. It's now early March where I live, nights are still around freezing, and yesterday I saw bees flying from one hive but not the other. When I put my ear to the quiet hive I heard a faint hum, and it feels light when I tilt it from the back. I fed both hives syrup in October and they seemed roughly the same weight then. I haven't opened either since. There are willows nearby that will flower in a couple of weeks and a lot of dandelions later in spring.\n\nBoth hives are standard wooden boxes with a single brood chamber and a mouse guard on the entrance. I have a bag of fondant and a spare feeder but no spare frames of honey. Last autumn the quiet hive had the larger population and I remember thinking it might be the stronger of the two going into winter. I don't have a mentor nearby and the nearest club meets once a month.",
        [
            "What is most likely going on with the quiet hive, and what should I do this week without harming it?",
            "How do I tell the difference between a colony that is about to starve and one that has already died?",
            "What should my plan be for both hives through April and May?",
        ],
    ),
    (
        "I'm planning a five-day trip down a lowland river in an open canoe with a friend. Neither of us has done a multi-day paddle before, though we've both done day trips. The river has two weirs with lock cuts, a stretch through farmland with barge traffic, and the last day and a half is tidal with a small port at the end. We'll camp on the bank. It's late September, so expect cold nights and possible frost, and days of maybe fifteen degrees. We can resupply in towns roughly once a day.\n\nWe have a two-person touring canoe, buoyancy aids, a small gas stove, a two-person tent, and dry bags for the sleeping kit. Neither of us has paddled in tidal water. We were told the lock keepers at both weirs will let us through if we arrive during working hours, but we don't know what those are. The friend I'm going with is a strong swimmer; I'm an average one.",
        [
            "What are the main risks on a trip like this and how would you plan around each of them?",
            "How should we handle the tidal section, and what information do we need before we get there?",
            "Give us a packing list organised by priority, keeping in mind we need to carry it all in one canoe.",
        ],
    ),
    (
        "I'm writing a short local-history piece about a decommissioned lighthouse on our coast. It was built in the 1860s with a first-order Fresnel lens and an oil lamp, converted to acetylene with a sun valve around 1920, electrified in the 1950s, and fully automated in 1988, when the last keeper left. There was a fog signal house with compressed-air horns added in the 1890s. The tower is now maintained by a volunteer trust and opens to visitors in summer. I'd like the piece to explain what the keepers actually did and why the job disappeared.\n\nThe piece is for a community magazine with a general readership, about 1,200 words, and I have access to the trust's archive of keepers' logbooks from the 1890s to the 1960s, a few photographs, and one recorded interview with the last keeper's daughter. I'd like to avoid the usual romantic tone and focus on the work itself. The editor has asked for at least one concrete anecdote per section.",
        [
            "Draft an outline for the piece, with a sentence or two describing what each section should cover.",
            "Explain, for a general reader, what each of the technical changes I listed meant for the keepers' daily work.",
            "What details or records should I look for in the local archive to make the piece vivid and accurate?",
        ],
    ),
    (
        "I run a small coffee stall at a weekend market. I buy roasted beans in five-kilogram bags from a local roaster once a fortnight, grind them all on Friday evening so Saturday and Sunday go faster, and brew with a batch filter machine. Customers have started saying the coffee tastes flat compared to a few months ago, when I was buying one-kilogram bags weekly. The roaster hasn't changed anything and I'm using the same beans, same grind setting, and same machine.\n\nThe ground coffee sits in the original bags, folded over and clipped, in a cupboard at the stall, which gets warm in the afternoon. I serve about 150 cups a weekend, so a five-kilogram bag lasts two weekends. I switched to the larger bags because they're about fifteen per cent cheaper per kilo. I have a small burr grinder at home but it's slow, maybe a minute per 60 grams, and the stall has no power for a grinder.",
        [
            "What is the most likely cause of the flat taste, and what would you change about my buying and grinding routine?",
            "How could I keep the Saturday morning workflow fast without pre-grinding everything the night before?",
            "Explain to me, as if I were a customer, why freshness matters so much for coffee.",
        ],
    ),
]

CHAT_LONG_INSTRUCTIONS = [
    "Answer using only the information in the passage above, and say so if the passage doesn't cover something.",
    "Give a clear answer in a short paragraph, then list the key points as bullets.",
    "Answer as if explaining to a curious teenager.",
    "Be concise and specific; quote the passage where it helps.",
    "Answer in no more than 150 words.",
]

# --------------------------------------------------------------------------
# rag_4k
# --------------------------------------------------------------------------
RAG_SYSTEM_VARIANTS = [
    "You are a helpful assistant. Answer the user's question using the retrieved passages below. Cite passages by their number in square brackets. If the passages do not contain the answer, say so.",
    "You answer questions strictly from the provided context. Reference the passages you use as [1], [2], etc. Do not use outside knowledge.",
    "Use the following search results to answer the question at the end. Be concise, cite sources by number, and note any disagreement between sources.",
]

RAG_QUESTION_SUFFIX = [
    "",
    " Answer in a few sentences.",
    " Give the answer, then list which passages supported it.",
    " If several passages are relevant, combine them.",
]

# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------
SUMMARIZE_INSTRUCTIONS = [
    "Summarize the article above in three to five bullet points.",
    "Write a one-paragraph summary of the article above for a general reader.",
    "Give me a two-sentence summary followed by a list of the key facts mentioned.",
    "Summarize the article above in under 120 words, preserving the most important specifics (names, numbers, dates).",
    "Produce an executive summary of the text above: a headline, then four short bullets.",
    "Summarize the article above, then suggest a title for it.",
]

# --------------------------------------------------------------------------
# code
# --------------------------------------------------------------------------
CODE_WRITE_TASKS = [
    ("Python", "Write a function `parse_duration(s: str) -> int` that accepts strings like `\"1h30m\"`, `\"45s\"`, `\"2h\"`, `\"1h 2m 3s\"`, or `\"90m\"` and returns the total number of seconds as an integer. Units are h, m, and s; components may appear in any order but each at most once; whitespace between components is optional."),
    ("Python", "Write a function `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` that takes a list of closed integer intervals, possibly overlapping and unsorted, and returns the minimal list of disjoint intervals that covers the same points, sorted by start. Adjacent intervals like (1, 3) and (4, 6) should be merged too, since they cover consecutive integers."),
    ("Python", "Write a small LRU cache class `LRU(capacity: int)` with `get(key)` and `put(key, value)` methods, both O(1), without using `functools.lru_cache` or `OrderedDict`. `get` returns `None` for a missing key. On `put` when the cache is full, evict the least recently used entry. Both `get` and `put` count as a use."),
    ("Python", "Write a generator `chunked(iterable, n)` that yields lists of at most `n` items from any iterable (including infinite ones and single-pass iterators), without materialising the whole input. The final chunk may be shorter. Raise `ValueError` if `n < 1`."),
    ("Rust", "Write a function `fn top_k(words: &[&str], k: usize) -> Vec<(String, usize)>` that returns the `k` most frequent words with their counts, most frequent first; break ties alphabetically. Use only the standard library. Include a couple of unit tests."),
    ("Rust", "Implement `fn parse_kv(line: &str) -> Result<(String, String), ParseError>` that splits a line of the form `key = value` on the first `=`, trims whitespace from both sides, rejects empty keys, and strips one pair of surrounding double quotes from the value if present. Define a small `ParseError` enum with variants for missing `=` and empty key, and implement `Display` for it."),
    ("Rust", "Write a `RingBuffer<T>` with fixed capacity given at construction, `push(&mut self, item: T)` that overwrites the oldest element when full, `len()`, `is_empty()`, and an `iter()` returning items from oldest to newest. Avoid unsafe code. Add tests that cover wrap-around."),
    ("JavaScript", "Write a function `debounce(fn, waitMs)` that returns a wrapped function which delays calling `fn` until `waitMs` milliseconds have passed without another call. The wrapper should pass through the latest arguments and `this`, and expose a `cancel()` method. Do not use any library."),
    ("JavaScript", "Write a function `groupBy(items, keyFn)` that returns a `Map` from each key produced by `keyFn(item)` to the array of items with that key, preserving original order within each group. Then show how you'd use it to group a list of `{name, city}` objects by city."),
    ("Go", "Write a function `func Retry(ctx context.Context, attempts int, base time.Duration, op func() error) error` that calls `op` up to `attempts` times with exponential backoff starting at `base` (doubling each time, capped at 10 seconds), returns nil on the first success, stops early if the context is cancelled, and returns the last error otherwise."),
    ("Go", "Write a function `func TopN(counts map[string]int, n int) []string` that returns the `n` keys with the largest counts in descending order of count, breaking ties by key ascending. Handle `n` larger than the map size."),
    ("SQL", "Given tables `orders(id, customer_id, placed_at, total_cents)` and `customers(id, name, country)`, write a query that returns, for each country, the number of distinct customers who placed at least one order in 2025 and the average order total in that country for that year, sorted by the customer count descending. Explain any assumptions about NULLs."),
    ("Bash", "Write a bash script that takes a directory as its only argument and prints, for each immediate subdirectory, its name and the total size of all files inside it in megabytes, sorted largest first. It should handle directory names with spaces and exit with a non-zero code and a message if the argument is not a directory."),
]

CODE_FIX_TASKS = [
    ("Python", "This function is supposed to return the median of a non-empty list of numbers, but it gives the wrong answer for lists with an even number of elements and crashes on a list of length one. Find the bugs and return a corrected version.",
     "def median(xs):\n    xs = sorted(xs)\n    n = len(xs)\n    mid = n / 2\n    if n % 2 == 1:\n        return xs[mid]\n    else:\n        return (xs[mid - 1] + xs[mid]) / 2\n"),
    ("Python", "This is meant to count word frequencies in a text, ignoring case and punctuation, and return the top 3 words. It doesn't behave as intended. Identify every problem you see and provide a fixed version.",
     "import string\n\ndef top_words(text):\n    counts = {}\n    for word in text.split(\" \"):\n        word = word.strip(string.punctuation)\n        if word in counts:\n            counts[word] = 1\n        else:\n            counts[word] += 1\n    top = sorted(counts, key=counts.get)[:3]\n    return top\n"),
    ("Python", "The following function should read a file of `key=value` lines into a dictionary, skipping blank lines and lines beginning with `#`. Users report it silently drops some values and sometimes raises. Fix it and explain each change briefly.",
     "def load_config(path):\n    cfg = {}\n    f = open(path)\n    for line in f.readlines():\n        if line.startswith('#') or line == '':\n            continue\n        key, value = line.split('=')\n        cfg[key] = value\n    return cfg\n"),
    ("Python", "This async function should fetch all URLs concurrently and return their bodies in the same order as the input. It runs them one at a time. Fix it so the requests actually overlap, keeping the ordering guarantee.",
     "import asyncio\n\nasync def fetch_all(session, urls):\n    results = []\n    for url in urls:\n        async with session.get(url) as resp:\n            results.append(await resp.text())\n    return results\n"),
    ("Rust", "This function should return the index of the first element greater than or equal to `target` in a sorted slice, or `None` if there is none. It sometimes loops forever and sometimes returns the wrong index. Fix it.",
     "fn lower_bound(xs: &[i32], target: i32) -> Option<usize> {\n    let mut lo = 0usize;\n    let mut hi = xs.len();\n    while lo < hi {\n        let mid = (lo + hi) / 2;\n        if xs[mid] < target {\n            lo = mid;\n        } else {\n            hi = mid;\n        }\n    }\n    if lo < xs.len() { Some(lo) } else { None }\n}\n"),
    ("Rust", "This does not compile. The intent is to remove all even numbers from the vector in place and then print the survivors. Explain why the borrow checker rejects it and give a version that compiles and does what was intended.",
     "fn main() {\n    let mut v = vec![1, 2, 3, 4, 5, 6];\n    for (i, x) in v.iter().enumerate() {\n        if x % 2 == 0 {\n            v.remove(i);\n        }\n    }\n    println!(\"{:?}\", v);\n}\n"),
    ("JavaScript", "This is supposed to log 0, 1, 2 after one, two, and three seconds. Instead it logs 3 three times. Explain why and give two different fixes.",
     "for (var i = 0; i < 3; i++) {\n  setTimeout(function () {\n    console.log(i);\n  }, (i + 1) * 1000);\n}\n"),
    ("JavaScript", "This function should return the sum of the `price` field across an array of items, treating missing or non-numeric prices as zero. It returns strings like `\"012.5\"` for some inputs and `NaN` for others. Fix it.",
     "function totalPrice(items) {\n  let total = 0;\n  for (const item in items) {\n    total += item.price || 0;\n  }\n  return total;\n}\n"),
    ("Go", "This is meant to start one goroutine per item, collect the results, and return them. It deadlocks. Explain why and fix it, preserving the order of results.",
     "func process(items []string) []string {\n    results := make(chan string)\n    for _, it := range items {\n        go func() {\n            results <- strings.ToUpper(it)\n        }()\n    }\n    var out []string\n    for range items {\n        out = append(out, <-results)\n    }\n    close(results)\n    return out\n}\n"),
    ("SQL", "This query is supposed to list every customer with the total they have spent, including customers who have never ordered (who should show 0). It omits those customers and double-counts for some others when there is also an `order_items` join elsewhere in the report. Fix the query and explain.",
     "SELECT c.name, SUM(o.total_cents) AS spent\nFROM customers c, orders o\nWHERE c.id = o.customer_id\nGROUP BY c.name;\n"),
]

CODE_CONSTRAINTS = [
    "Include a short docstring or comment explaining the approach.",
    "Add three or four test cases covering edge cases.",
    "Do not use any third-party libraries.",
    "Prefer clarity over cleverness; this will be read by junior developers.",
    "Keep the time complexity as low as you reasonably can and state what it is.",
    "Handle invalid input explicitly rather than letting it raise an unrelated error.",
    "Explain any tradeoffs you made in a sentence or two after the code.",
    "Match the surrounding code style: four-space indentation, descriptive names, no single-letter variables except loop indices.",
    "Return only the code and a brief explanation; no preamble.",
    "Assume inputs may be large (millions of items), so avoid quadratic behaviour.",
]

CODE_CONTEXTS = [
    "",
    "",
    "This is for an internal command-line tool used by our operations team.",
    "This is part of a data-processing pipeline that runs nightly on a few gigabytes of logs.",
    "This will go into a small web service that handles a few hundred requests a second.",
    "This is for a teaching example, so correctness and readability matter more than speed.",
    "We are on a fairly old runtime version, so avoid very new language features.",
    "This code is in a library that other teams depend on, so keep the public signature exactly as given.",
]

# Longer background paragraphs, added when the sampled target length is high.
CODE_BACKGROUND = [
    "Background: the existing implementation was written quickly during an incident and has never been cleaned up. It works for the common case but we have had two production bugs traced to it in the last quarter, both involving inputs at the boundaries (empty input, a single element, and very large values). The team has agreed that the replacement should be boring and obviously correct rather than clever, and that it should come with tests we can run in CI.",
    "Background: this runs inside a request handler, so anything that blocks for a long time or allocates proportionally to the input size will show up as latency for users. We measure p99 latency and the budget for this piece is a few milliseconds for typical inputs of a few thousand items. It is fine to trade a little memory for speed, but please point out where you did so.",
    "Background: the code is called from several places with slightly different expectations, and a previous attempt to fix it broke one of the callers because the return type changed. Please keep the signature and return type exactly as written, and if you think the interface itself is wrong, say so in a note rather than changing it. We can do the interface change as a separate follow-up.",
    "Background: we have a mix of experience levels on the team, and this file is often the first one new hires read. Comments that explain why, not what, are welcome. We follow the standard formatter for the language and run a linter in CI with the default rules; anything that would trip the linter will be sent back, so please keep names descriptive and functions short.",
    "Background: input comes from an external system that we do not control, so it is occasionally malformed: trailing whitespace, unexpected casing, duplicate entries, and the odd null. The current behaviour on bad input is to crash the whole batch, which is worse than skipping the bad record. Where you make a decision about how to handle bad input, log or comment it so operations can find it later.",
    "Background: this was ported from another language a few years ago and still shows it: manual index loops where the standard library has a direct equivalent, string concatenation in loops, and error handling by returning magic values. Part of the goal here is to make it idiomatic for the language it is in now, not just correct. Idiomatic error handling is particularly important.",
]


# --------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------
def _fill(paragraphs, target_tokens, start=0, min_paras=1):
    """Consecutive paragraphs from `start` until the token estimate reaches the target."""
    out, toks, i = [], 0, start
    while i < len(paragraphs) and (len(out) < min_paras or toks < target_tokens):
        out.append(paragraphs[i])
        toks += est_tokens(paragraphs[i])
        i += 1
    return out


def gen_chat_short(rng):
    q = rng.choice(SHORT_QUESTIONS)
    text = rng.choice(SHORT_PREFIX) + q + rng.choice(SHORT_SUFFIX)
    return [{"role": "user", "content": text}]


def gen_chat_long(rng):
    lo, hi = RANGES["chat_long"]
    target = rng.randint(lo, hi)
    if rng.random() < 0.35:
        ctx, qs = rng.choice(SCENARIOS)
        q = rng.choice(qs)
        text = ctx + "\n\n" + q
        if est_tokens(text) < target - 80:
            others = [x for x in qs if x != q]
            extra = rng.choice(others)
            text += " Also: " + extra[0].lower() + extra[1:]
        return [{"role": "user", "content": text}]
    title, paras, qs = rng.choice(DOCS)
    start = rng.randint(0, max(0, len(paras) - 5))
    chosen = _fill(paras, target - 60, start=start, min_paras=2)
    q = rng.choice(qs)
    lead = rng.choice([
        'Here is an excerpt from an article titled "%s":' % title,
        'I\'m reading a piece called "%s". This is the relevant part:' % title,
        'Context (from "%s"):' % title,
        "Passage:",
    ])
    text = lead + "\n\n" + "\n\n".join(chosen) + "\n\n" + q + " " + rng.choice(CHAT_LONG_INSTRUCTIONS)
    return [{"role": "user", "content": text}]


def gen_rag_4k(rng):
    lo, hi = RANGES["rag_4k"]
    target = rng.randint(lo, hi)
    doc_ids = list(range(len(DOCS)))
    rng.shuffle(doc_ids)
    title, paras, qs = DOCS[doc_ids[0]]
    q = rng.choice(qs)
    # Answer passage: 3-paragraph window around the question's likely paragraph
    # (questions are ordered roughly like the paragraphs).
    qi = qs.index(q)
    center = round(qi * (len(paras) - 1) / max(1, len(qs) - 1))
    a0 = max(0, min(center - 1, len(paras) - 3))
    answer_chunk = paras[a0:a0 + 3]
    passages = [(title, answer_chunk)]
    toks = est_tokens(" ".join(answer_chunk))
    for d in doc_ids[1:]:
        if toks >= target:
            break
        t, p, _ = DOCS[d]
        n = rng.choice([2, 2, 3])
        s = rng.randint(0, len(p) - n)
        chunk = p[s:s + n]
        passages.append((t, chunk))
        toks += est_tokens(" ".join(chunk))
    rng.shuffle(passages)
    parts = ["[%d] %s\n%s" % (i, t, "\n".join(chunk)) for i, (t, chunk) in enumerate(passages, 1)]
    user = "Retrieved passages:\n\n" + "\n\n".join(parts) + "\n\nQuestion: " + q + rng.choice(RAG_QUESTION_SUFFIX)
    return [
        {"role": "system", "content": rng.choice(RAG_SYSTEM_VARIANTS)},
        {"role": "user", "content": user},
    ]


def gen_code(rng):
    lo, hi = RANGES["code"]
    target = rng.randint(lo, hi)
    if rng.random() < 0.5:
        lang, task = rng.choice(CODE_WRITE_TASKS)
        body = "%s\n\nLanguage: %s." % (task, lang)
    else:
        lang, task, snippet = rng.choice(CODE_FIX_TASKS)
        body = "%s\n\n```%s\n%s```" % (task, lang.lower(), snippet)
    ctx = rng.choice(CODE_CONTEXTS)
    if ctx:
        body = ctx + "\n\n" + body
    bg = CODE_BACKGROUND[:]
    rng.shuffle(bg)
    while bg and est_tokens(body) < target * 0.9 - 130:
        body = bg.pop() + "\n\n" + body
    pool = CODE_CONSTRAINTS[:]
    rng.shuffle(pool)
    cons, text = [], body
    # Code tokenizes denser than prose (est_tokens undercounts ~10%).
    while pool and est_tokens(text) < target * 0.9:
        cons.append(pool.pop())
        text = body + "\n\nRequirements:\n" + "\n".join("- " + c for c in cons)
    return [{"role": "user", "content": text}]


def gen_summarize(rng):
    lo, hi = RANGES["summarize"]
    target = rng.randint(lo, hi)
    doc_ids = list(range(len(DOCS)))
    rng.shuffle(doc_ids)
    title, paras, _ = DOCS[doc_ids[0]]
    start = rng.randint(0, 1)
    chosen = _fill(paras, target - 40, start=start, min_paras=4)
    article = "# %s\n\n%s" % (title, "\n\n".join(chosen))
    if est_tokens(article) < target - 150:
        t2, p2, _ = DOCS[doc_ids[1]]
        extra = _fill(p2, target - est_tokens(article) - 40)
        article += "\n\n## Related: %s\n\n%s" % (t2, "\n\n".join(extra))
    text = article + "\n\n" + rng.choice(SUMMARIZE_INSTRUCTIONS)
    return [{"role": "user", "content": text}]


GENERATORS = {
    "chat_short": gen_chat_short,
    "chat_long": gen_chat_long,
    "rag_4k": gen_rag_4k,
    "code": gen_code,
    "summarize": gen_summarize,
}

MIXED_WEIGHTS = [("chat_short", 30), ("chat_long", 25), ("code", 20), ("summarize", 15), ("rag_4k", 10)]


def make_request(workload, seed, index):
    """Deterministic request `index` of `workload` for `seed` -> {"kind", "index", "messages"}."""
    rng = random.Random("%d:%s:%d" % (seed, workload, index))
    kind = workload
    if workload == "mixed":
        names = [n for n, _ in MIXED_WEIGHTS]
        weights = [w for _, w in MIXED_WEIGHTS]
        kind = rng.choices(names, weights=weights, k=1)[0]
    return {"kind": kind, "index": index, "messages": GENERATORS[kind](rng)}
