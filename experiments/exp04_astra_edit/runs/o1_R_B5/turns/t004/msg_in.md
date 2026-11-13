OUTPUT:
```
WORK: C:\code\qualiusp2026\experiments\exp04_astra_edit\runs\o1_R_B5\work | NOTES exists: True

n tensors: 105
['bert.encoder.layer.5.attention.output.LayerNorm.bias', 'bert.encoder.layer.5.attention.output.LayerNorm.weight', 'bert.encoder.layer.5.attention.output.dense.bias', 'bert.encoder.layer.5.attention.output.dense.weight', 'bert.encoder.layer.5.attention.self.key.bias', 'bert.encoder.layer.5.attention.self.key.weight', 'bert.encoder.layer.5.attention.self.query.bias', 'bert.encoder.layer.5.attention.self.query.weight', 'bert.encoder.layer.5.attention.self.value.bias', 'bert.encoder.layer.5.attention.self.value.weight', 'bert.encoder.layer.5.intermediate.dense.bias', 'bert.encoder.layer.5.intermediate.dense.weight', 'bert.encoder.layer.5.output.LayerNorm.bias', 'bert.encoder.layer.5.output.LayerNorm.weight', 'bert.encoder.layer.5.output.dense.bias', 'bert.encoder.layer.5.output.dense.weight', 'bert.pooler.dense.bias', 'bert.pooler.dense.weight']
n questions: 200 types: Counter({'bridge': 161, 'comparison': 39})
paragraph counts: Counter({10: 199, 4: 1}) total: 1994

Q: Who was the director of the film whose success, along with that of An American Tail and The Land Before Time, prompted Steven Spielberg to establish his own animation studio? | A: Robert Zemeckis | bridge
  0 An American Tail: Fievel Goes West. An American Tail: Fievel Goes West (also known as An American Tail II: Fie
  0 Morris Sullivan. Morris Francis Sullivan (December 8, 1916 – August 24, 2008) was an American businessman who 
  0 Event movie. An event movie is a film whose release itself is considered a major event, such as an anticipated
  0 Amblimation. Amblimation was the animation production arm of Steven Spielberg's Amblin Entertainment. The stud
  0 Mohanlal filmography. []Mohanlal] is an Indian actor, producer, and singer who has starred in both blockbuster
  0 Universal Animation Studios. Universal Animation Studios (formerly known as Universal Cartoon Studios) is an A
  0 Balto (film). Balto is a 1995 American animated epic drama adventure film directed by Simon Wells, produced by
  1 An American Tail. An American Tail is a 1986 American animated musical adventure family comedy-drama film dire
  1 Who Framed Roger Rabbit. Who Framed Roger Rabbit is a 1988 American live-action/animated fantasy film directed
  0 Pinky, Elmyra &amp; the Brain. Steven Spielberg Presents Pinky, Elmyra & the Brain is the retooling of the ani

Q: What town holds a celebration the name of which literally means "apple slices"? | A: Tyler | bridge
  0 Iekei Ramen. Iekei Ramen (家系ラーメン , Iekei Rāmen ) is a ramen noodle bar chain originating from a ramen bar call
  1 Danebod. Danebod (pronounced "DAHN-a-bo", or "DAN-a-bo" ) is a historic district at the south edge of the rail
  1 Æbleskiver. Æbleskiver "Pancake Puffs" (] , [singular: "æbleskive"]), also called "appleskives" are traditiona
  0 Indian Trail, North Carolina. Indian Trail is a suburban town in Union County, North Carolina, United States. 
  0 Shuizhu. Shuizhuroupian () is a Chinese dish which originated from the cuisine of Sichuan province and the nam
  0 Haijiao. Haijiao (海礁), also known as Tong Island (童岛), Taibujiao (泰簿礁) or Taijiao (泰礁), is located at 30°44'6"
  0 Ayden, North Carolina. Ayden is a town in Pitt County, North Carolina, United States. The population was 4,932
  0 Rajas con crema. Rajas con crema is the name given to a Mexican dish consisting of sliced poblano pepper with 
  0 Princeton, North Carolina. Princeton is a town in Johnston County, North Carolina, United States. Each May the
  0 Itarsi. Itarsi is a town in Madhya Pradesh, India in Hoshangabad District. Itarsi is a key hub for agricultura
para token len: mean 120.3, median 108, max 618
edited(): {'track': 'R', 'heads_removed': [], 'neurons_removed': 0, 'layers_removed': [], 'scalings': 0, 'cost': {'encoder_gflops_per_question': 39.2620032, 'encoder_params': 10646784, 'removed': {'heads': 0, 'neurons': 0, 'layers': []}}}

```
Passes left: 5/5. Turn 3/40.