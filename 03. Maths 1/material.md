# Конспект

Вероятность.

Для чего нам нужны вероятности, когда у нас итак есть много других математических инструментов?
У нас есть математический анализ для работы с функциями на бесконечных интервалах. У нас есть алгебра для решений систем уравнений и т.д. Основная проблема заключается в том, что мы живем в хаотической Вселенной, где мы не можем измерить все вещи точно. Случайные события, неучтенный данные влияют на наши эксперименты, искажая их. Т.е. везде возникает некая неуверенность в результатах и ошибки. Вот тут выходит на передний план теория вероятностей и статистика.
Есть много определение вероятности, рассмотрим частное. Будем подбрасывать монетку и получать два результата: орёл или решку. Подбросим ﻿
1000
1000﻿ раз, получим: ﻿
600
600﻿ орлов и ﻿
400
400﻿ решек. Тогда вероятность выпадения орла равна ﻿
600
1000
=
0.6
1000
600
​	
 =0.6﻿ , а вероятность выпадения решки  —  ﻿
400
1000
=
0.4
1000
400
​	
 =0.4﻿ . Если говорить формально, то  вероятность события ﻿
A
A﻿ равна количеству появлений этого события ﻿
N
A
N 
A
​	
 ﻿ по отношению к количеству всех событий ﻿
N
N﻿  .
﻿
P
(
A
)
=
N
A
N
P(A)= 
N
N 
A
​	
 
​	
 
​	
 ﻿
Например, мы по очереди подбрасываем две игральные кости. Пусть событие ﻿
A
=
{
"
в
 
с
у
м
м
е
 
н
а
 
к
о
с
т
я
х
 
в
ы
п
а
л
о
 
10
"
}
A={"в сумме на костях выпало 10"}﻿ , найдите ﻿
P
(
A
)
=
?
P(A)=?﻿ . Рассмотрим все возможности, когда сумма будет равно
﻿
10
10﻿ : ﻿
(
4
,
6
)
,
 
(
5
,
5
)
,
 
(
4
,
6
)
.
(4,6), (5,5), (4,6).﻿ Т.е. ﻿
N
A
=
3
,
 
N
=
6
∗
6
=
36.
N 
A
​	
 =3, N=6∗6=36.﻿
Тогда ﻿
P
(
A
)
=
3
36
=
1
12
.
P(A)= 
36
3
​	
 = 
12
1
​	
 .﻿

Условные вероятности.

Часто нам необходимо знать не только вероятность какого-то события, а вероятность события, когда уже что-то случилось. Например, событие "завтра днем будет дождь". Если вы задаетесь этим вопросом сегодня вечером, то вероятность может быть одна. Однако, если вы подумаете об этом завтра утром, то вероятность может быть другая. Например, если с утра облачно, то вероятность дождя вечером повысится (например). Это и есть условная вероятность. Обозначается ﻿
P
(
A
∣
B
)
P(A∣B)﻿  — вероятность события  ﻿
A
A﻿ при условии того, что событие ﻿
B
B﻿ сбылось. Рассмотрим ещё примеры:
Какова вероятность дождя, если гремит гром?
Какова вероятность дождя, если сейчас солнечно?



Немного упрощённое изображение в виде диаграмм Эйлера, но оно передаёт смысл. По диаграмме можно понять, что ﻿
P
(
д
о
ж
д
ь
∣
г
р
о
м
)
=
1
P(дождь∣гром)=1﻿ , т.е. когда идёт гром, то всегда будет дождь. А что насчёт ﻿
P
(
д
о
ж
д
ь
∣
с
о
л
н
е
ч
н
о
)
=
P
(
д
о
ж
д
ь
 
и
 
с
о
л
н
е
ч
н
о
)
P
(
с
о
л
н
е
ч
н
о
)
,
P(дождь∣солнечно)= 
P(солнечно)
P(дождь и солнечно)
​	
 ,﻿
﻿
т
.
е
.
 
P
(
A
∣
B
)
=
P
(
A
,
B
)
P
(
B
)
т.е. P(A∣B)= 
P(B)
P(A,B)
​	
 ﻿
События ﻿
A
 
и
 
B
A и B﻿ называются  независимыми,  если ﻿
P
(
A
,
B
)
=
P
(
A
)
P
(
B
)
,
P(A,B)=P(A)P(B),﻿ или то же самое, что ﻿
P
(
A
)
=
P
(
A
∣
B
)
.
P(A)=P(A∣B).﻿ На данной диаграмме все события зависимы, т.к. если происходит одно, то можно сделать выводы о том, произойдет ли другое. Пример независимых событий: погода и количество приложений в твоем телефоне.
Основные свойства вероятности.

Вероятность невозможного события равно нулю: ﻿
P
(
∅
)
=
0
P(∅)=0﻿
Если событие ﻿
A
A﻿ включается в ﻿
B
B﻿ , т.е. если наступило событие ﻿
A
A﻿ , то событие ﻿
B
B﻿ точно наступило, то: ﻿
P
(
A
)
≤
P
(
B
)
P(A)≤P(B)﻿
Вероятность каждого события ﻿
A
A﻿ находится от ﻿
0
0﻿ до ﻿
1
1﻿ .
﻿
P
(
B
\
A
)
=
P
(
B
)
−
P
(
A
)
P(B\A)=P(B)−P(A)﻿  — событие, когда  ﻿
B
B﻿ наступило, а ﻿
A
A﻿  — нет. 
 Вероятность противоположного события к  ﻿
A
A﻿ равна: ﻿
P
(
A
‾
)
=
1
−
P
(
A
)
P( 
A
 )=1−P(A)﻿
﻿
P
(
A
+
B
)
=
P
(
A
)
+
P
(
B
)
−
P
(
A
B
)
=
0
P(A+B)=P(A)+P(B)−P(AB)=0﻿ , то события называют  несовместимыми  .

Формула Байеса.

Мы знаем, что ﻿
P
(
A
∣
B
)
=
P
(
A
B
)
P
(
B
)
P(A∣B)= 
P(B)
P(AB)
​	
 ﻿ . Давайте найдем вероятность другого события ﻿
P
(
B
∣
A
)
=
?
P(B∣A)=?﻿ По формуле условной вероятности получаем:
﻿
P
(
B
∣
A
)
=
P
(
B
A
)
P
(
A
)
P(B∣A)= 
P(A)
P(BA)
​	
 
​	
 ﻿
Но т.к. ﻿
P
(
A
B
)
=
P
(
B
A
)
P(AB)=P(BA)﻿ , ибо это одно и то же, а ﻿
P
(
A
B
)
=
P
(
A
∣
B
)
P
(
B
)
P(AB)=P(A∣B)P(B)﻿ , то получаем  формулу Байеса: 
﻿
P
(
B
∣
A
)
=
P
(
B
A
)
P
(
A
)
=
P
(
A
B
)
P
(
A
)
=
P
(
A
∣
B
)
P
(
B
)
P
(
A
)
.
P(B∣A)= 
P(A)
P(BA)
​	
 = 
P(A)
P(AB)
​	
 = 
P(A)
P(A∣B)P(B)
​	
 
​	
 .﻿
Формула Байеса помогает "переставить причину и следствие": найти вероятность того, что событие ﻿
B
B﻿ было вызвано причиной ﻿
A
A﻿ .

Формула полной вероятности.

Пусть ﻿
B
1
,
…
,
B
n
B 
1
​	
 ,…,B 
n
​	
 ﻿  — несовместимые события, т.е. те, которые не могут произойти одновременно. Например, вы на экзамене тянете только  ﻿
1
1﻿ билет из ﻿
15
15﻿ . Вам не может попасться два билета одновременно. Ещё дополнительное условие на эти события такое, что вместе они образуют все возможные исходы: ﻿
P
(
B
1
)
+
⋯
+
P
(
B
n
)
=
1
P(B 
1
​	
 )+⋯+P(B 
n
​	
 )=1﻿ . Это означает, что хотя бы одно событие да произойдет. Например, на том же экзамене вам в любом случае достанется один из билетов. Говорят, что событие ﻿
B
1
,
…
,
B
n
B 
1
​	
 ,…,B 
n
​	
 ﻿ образуют   полную группу несовместных событий.  
Пусть ещё есть событие ﻿
A
A﻿  —  например, вам в билете попалась тема
про условные вероятности. Рассмотрим вероятности ﻿
P
(
A
,
B
1
)
,
…
,
P
(
A
,
B
n
)
P(A,B 
1
​	
 ),…,P(A,B 
n
​	
 )﻿  —  т.е. вероятность того, что вам выпал
﻿
i
i﻿ -ый билет и тема про условные вероятности. Т.к. вам обязательно достанется какой-то билет, то:
﻿
P
(
A
)
=
P
(
A
,
B
1
)
+
⋯
+
P
(
A
,
B
n
)
P(A)=P(A,B 
1
​	
 )+⋯+P(A,B 
n
​	
 )
​	
 ﻿
Распишем каждую вероятность по правилу условной вероятности:
﻿
P
(
A
)
=
P
(
A
,
B
1
)
+
⋯
+
P
(
A
,
B
n
)
=
P
(
B
1
)
P
(
A
∣
B
1
)
+
⋯
+
P
(
B
n
)
P
(
A
∣
B
n
)
=
∑
i
=
1
n
P
(
B
i
)
P
(
A
∣
B
i
)
P(A)=P(A,B 
1
​	
 )+⋯+P(A,B 
n
​	
 )=P(B 
1
​	
 )P(A∣B 
1
​	
 )+⋯+P(B 
n
​	
 )P(A∣B 
n
​	
 )= 
i=1
∑
n
​	
 P(B 
i
​	
 )P(A∣B 
i
​	
 )﻿  — это   формула полной вероятности. 
С её помощью можно усовершенствовать формулу Байеса:
﻿
P
(
B
i
∣
A
)
=
P
(
B
i
)
P
(
A
∣
B
i
)
P
(
A
)
=
P
(
B
i
)
P
(
A
∣
B
i
)
∑
i
=
1
n
P
(
B
i
)
P
(
A
∣
B
i
)
P(B 
i
​	
 ∣A)= 
P(A)
P(B 
i
​	
 )P(A∣B 
i
​	
 )
​	
 = 
i=1
∑
n
​	
 P(B 
i
​	
 )P(A∣B 
i
​	
 )
P(B 
i
​	
 )P(A∣B 
i
​	
 )
​	
 
​	
 ﻿
Вероятности ﻿
P
(
B
i
∣
A
)
P(B 
i
​	
 ∣A)﻿ называют  апостериорными вероятностями  (т.е. вероятности, которые немного уточнены, т.к. произошло событие ﻿
A
A﻿ , а события ﻿
P
(
B
i
)
P(B 
i
​	
 )﻿  —   априорными вероятностями  (вероятности, которые известны до проведения эксперимента, связанного с ﻿
A
A﻿ ).

Наивный байесовский классификатор.

Воспользуемся предыдущими формулами для задачи классификации. Пусть у нас есть задача классификации текстов на ﻿
C
=
{
c
1
,
…
,
c
n
}
C={c 
1
​	
 ,…,c 
n
​	
 }﻿
классов. Что бы взять в качестве признаков? Например, какие слова
встречаются в текстах. Т.е. пусть во всех текстах всего ﻿
N
N﻿ уникальных
слов. Мы дадим каждому слову номер от ﻿
1
 
д
о
 
N
1 до N﻿ . Признаками каждого текста будет кол-во каждого слова, которое есть в тексте. Например, ﻿
2
2﻿ слова "дом", ﻿
3
3﻿ слова "кот", ﻿
0
0﻿ слов "аниме". Таким образом, каждый текст мы можем представить в виде вектора длины ﻿
N
:
(
w
1
,
.
.
.
,
w
N
)
N:(w 
1
​	
 ,...,w 
N
​	
 )﻿ , где ﻿
x
i
x 
i
​	
 ﻿  —  количество слов под номером ﻿
i
i﻿ в нашем тексте.
У нас есть много таких текстов, поэтому мы можем вычислить различные вероятности. Например:
Сколько есть текстов каждого класса ﻿
P
(
c
)
P(c)﻿
Какова вероятность встречи слова под номером ﻿
1
:
P
(
w
1
)
1:P(w 
1
​	
 )﻿
Сколько раз встречается слово под номером ﻿
2
2﻿ разных классах ﻿
P
(
w
2
∣
c
)
P(w 
2
​	
 ∣c)﻿ . Для этого берём все тексты, содержащие слово ﻿
w
2
w 
2
​	
 ﻿ . Смотрим, сколько из этих писем помечено классом ﻿
c
1
c 
1
​	
 ﻿ , сколько классом ﻿
c
2
c 
2
​	
 ﻿ и т.д.
В задаче же классификации у нас есть текст, состоящее из таких-то
слов, т.е. имеет признаки. Наша задача состоит в том, чтобы понять класс текста ﻿
c
c﻿ . Т.е. найти вероятности ﻿
P
(
c
∣
w
1
,
…
,
w
N
)
P(c∣w 
1
​	
 ,…,w 
N
​	
 )﻿ ,
перебрать все ﻿
c
c﻿ и выбрать из них тот класс, который даёт максимальную вероятность. Формально:
﻿
c
^
=
c
P
(
c
∣
w
1
,
…
w
N
)
,
г
д
е
 
c
^
c
^
 = 
c
argmax
​	
 P(c∣w 
1
​	
 ,…w 
N
​	
 ),где  
c
^
 ﻿  —  наше предсказание.
Преобразуем эту вероятность по формуле Байеса:
﻿
P
(
c
∣
w
1
,
…
,
w
N
)
=
P
(
c
,
w
1
,
…
w
N
)
P
(
w
1
,
…
,
w
N
)
P(c∣w 
1
​	
 ,…,w 
N
​	
 )= 
P(w 
1
​	
 ,…,w 
N
​	
 )
P(c,w 
1
​	
 ,…w 
N
​	
 )
​	
 
​	
 ﻿
Т.к. мы максимизируем по ﻿
c
c﻿ , а вероятности ﻿
w
1
,
…
w
N
w 
1
​	
 ,…w 
N
​	
 ﻿ остаются
для письма теми же, то их можно убрать из argmax.
﻿
P
(
c
,
w
1
,
…
,
w
N
)
=
P
(
c
)
P
(
w
1
,
…
,
w
N
∣
c
)
P(c,w 
1
​	
 ,…,w 
N
​	
 )=P(c)P(w 
1
​	
 ,…,w 
N
​	
 ∣c)
​	
 ﻿
Вероятность справа достаточно сложная. Давайте сделаем "наивное" предположение: все слова в текстах появляются независимо друг от друга. Т.е. если в тексте есть ﻿
2
2﻿ слова "котик", то количество "собак" может быть любым. В этом и заключается слово "наивный" в названии данного алгоритма.
﻿
P
(
c
,
w
1
,
…
,
w
N
)
=
P
(
c
)
P
(
w
1
∣
c
)
…
P
(
w
N
∣
c
)
P(c,w 
1
​	
 ,…,w 
N
​	
 )=P(c)P(w 
1
​	
 ∣c)…P(w 
N
​	
 ∣c)
​	
 ﻿
Все вероятности справа мы можем подсчитать по обучающей выборке. Тогда получаем итоговый ответ алгоритма:
﻿
c
^
=
c
P
(
c
)
P
(
w
1
∣
c
)
.
.
.
P
(
w
N
∣
c
)
c
^
 = 
c
argmax
​	
 P(c)P(w 
1
​	
 ∣c)...P(w 
N
​	
 ∣c)
​	
 ﻿

Распределения вероятностей.

Что это такое? Это закон, который говорит нам, какая вероятность у
каждого результата эксперимента, и всё это выражено математической функцией. Например, распределение результатов игральной кости  —  это  равномерное распределение с вероятностью  ﻿
1
6
.
6
1
​	
 .﻿

Результат случайного эксперимента так же называют  случайной 
 величиной  . Говорят, что случайная величина подчиняется какому-то
распределению вероятностей. Например, если ﻿
X
X﻿  —  результат
подброса игральной кости, то пишут:
﻿
X
∼
U
n
i
f
o
r
m
(
1
/
6
)
X∼Uniform(1/6)
​	
 ﻿

Непрерывные и дискретные распределения.

До этого мы обсуждали только дискретные распределения, т.е. результат эксперимента всегда берётся из какого-то конечного множества. Но в реальности, многие физические измерения выдают действительное число на числовой прямой. Такие распределения называются непрерывными. Для них нет вероятности выпадения определённого результата, зато есть
плотность распределения ﻿
p
(
x
)
p(x)﻿  —  это некая функция, которая оценивает вероятность выпадения данного числа. Она может быть больше единицы.
Рассмотрим одно из непрерывных распределений  —  нормальное.

Оно выдаёт дробные числа, которые как правило колеблются около среднего. На картинке выше, среднее - это ﻿
0
0﻿ . У этого распределения есть ещё один параметр  —  это дисперсия распределения. Этот параметр показывает, крутой склон у этого распределения. На картинке навскидку дисперсия равна ﻿
1
1﻿ . Пишут, что ﻿
X
∼
N
(
0
;
1
)
X∼N(0;1)﻿ .

Плотность нормального распределения достаточно сложная и её необязательно запоминать:
﻿
p
(
x
)
=
1
2
π
σ
2
exp
⁡
(
−
(
x
−
μ
)
2
2
σ
2
)
p(x)= 
2πσ 
2
 
​	
 
1
​	
 exp(− 
2σ 
2
 
(x−μ) 
2
 
​	
 )﻿ , где ﻿
μ
μ﻿  — среднее, а  ﻿
σ
σ﻿  — дисперсия. 
В данном случае среднее и дисперсия -- параметры распределения и известны заранее. Но это не всегда так, поэтому их вычисляют по следующим формулам:
Для дискретных:
﻿
E
(
X
)
=
∑
x
P
(
X
=
x
)
E(X)=∑xP(X=x)﻿  —  мат.ожидание (среднее)
﻿
D
(
X
)
=
E
X
2
−
(
E
X
)
2
=
∑
x
2
P
(
X
=
x
)
−
(
E
X
)
2
D(X)=EX 
2
 −(EX) 
2
 =∑x 
2
 P(X=x)−(EX) 
2
 ﻿
Для непрерывных:
﻿
E
(
X
)
=
∫
x
p
(
x
)
d
x
E(X)=∫xp(x)dx﻿
﻿
D
(
X
)
=
E
X
2
−
(
E
X
)
2
=
∫
x
2
p
(
x
)
d
x
−
(
E
X
)
2
D(X)=EX 
2
 −(EX) 
2
 =∫x 
2
 p(x)dx−(EX) 
2
 ﻿

Пока интегралы вы не изучали, но потом поймёте  :) 

Распределение Пуассона.

Результатами распределения Пуассона являются натуральные числа. Поэтому оно дискретное:

Как правило, распределение Пуассона моделирует эксперимент, который вычисляет кол-во событий, произошедших за какое-то конкретное время. Эти события должны происходить с какой-то средней интенсивностью и независимо друг от друга. Например, количество машин, которое проедет за час через ваш двор.
У него есть параметр ﻿
λ
λ﻿ , который является его средним, а также
дисперсией:
﻿
X
∼
P
o
i
s
s
o
n
(
λ
)
X∼Poisson(λ)﻿
﻿
P
(
X
=
k
)
=
λ
k
k
!
e
−
λ
P(X=k)= 
k!
λ 
k
 
​	
 e 
−λ
 ﻿

Метод максимального правдоподобия.

Пусть на входе мы имеем какую-то случайную выборку чисел ﻿
X
X﻿ . Нам хотелось бы определить, из какого они распределения. Если мы это узнаем, то можем, например, генерировать больше данных для обучения. Обычно, если построить гистограмму, то можно на глаз оценить, к какому семейству распределений оно относится (нормальное, экспоненциальное, пуассона и т.д.). Осталось найти параметры этих распределений, которые обозначим за ﻿
θ
θ﻿ .

Так как мы знаем, к какому семейству относится наше распределение, то мы можем записать вероятность или плотность вероятности ﻿
p
(
x
)
p(x)﻿ , как функцию, в которой кроме ﻿
x
x﻿ будет ещё фигурировать ﻿
θ
θ﻿ . Суть метода заключается в следующем:
Находим вероятность нашей выборки: ﻿
P
(
X
∣
θ
)
=
∏
i
=
1
n
P
(
x
i
∣
θ
)
P(X∣θ)= 
i=1
∏
n
​	
 P(x 
i
​	
 ∣θ)﻿  —  функция правдоподобия.
Давайте будем искать такой ﻿
θ
θ﻿ , при котором эта вероятность максимальна. Почему так делают?
Приведём пример. Пусть у нас есть игральная кость, в которой одна из граней утяжелена, т.е. она выпадает чаще. Пусть у вас есть ﻿
1000
1000﻿ бросков этой кости, и вас просят на основе этих данных понять, какая грань утяжелена. Естественно, что вы ответите,
что та утяжелена та грань, которая чаще всего выпала. Это и есть метод максимального правдоподобия. Параметром в нашем странном распределении является номер грани, которая утяжелена. Если больше всего выпадала грань " ﻿
3
3﻿ ", то вероятность того, что она утяжелена намного больше, чем то, что утяжелена грань " ﻿
2
2﻿ "'. Поэтому мы перебираем все варианты ﻿
θ
θ﻿ и находим из них ту, которая даёт максимальную вероятность:
﻿
θ
^
=
θ
∏
i
=
1
n
P
(
x
i
∣
θ
)
θ
^
 = 
θ
argmax
​	
  
i=1
∏
n
​	
 P(x 
i
​	
 ∣θ)﻿
Если ﻿
θ
θ﻿  —  вещественный параметр, то обычно используют производную функции, чтобы найти минимум. Совет: можно искать не минимум произведения, а логарифм от этого минимума. Так вы будете оперировать с меньшими числами и производную суммы легче посчитать, чем производную от произведения:
﻿
θ
^
=
θ
log
⁡
(
∏
i
=
1
n
P
(
x
i
∣
θ
)
)
=
θ
∑
i
=
1
n
log
⁡
(
P
(
x
i
∣
θ
)
)
θ
^
 = 
θ
argmax
​	
 log( 
i=1
∏
n
​	
 P(x 
i
​	
 ∣θ))= 
θ
argmax
​	
  
i=1
∑
n
​	
 log(P(x 
i
​	
 ∣θ))
​	
 ﻿

Связь MSE и ММП.

Рассмотрим, классическую регрессионную модель:
﻿
y
i
=
w
x
i
+
ϵ
,
г
д
е
 
ϵ
∼
N
(
0
;
1
)
y 
i
​	
 =wx 
i
​	
 +ϵ,где ϵ∼N(0;1)﻿
Другими словами, у нас есть данные какого-то измерения, которые лежат на линии (гиперплоскости в многомерном случае) с каким-то шумом ﻿
ϵ
ϵ﻿ , который подчиняется стандартному нормальному распределению ﻿
N
(
0
;
1
)
N(0;1)﻿ .
Давайте с помощью метода максимального правдоподобия найдём параметры ﻿
w
w﻿ . Что такое ﻿
y
i
y 
i
​	
 ﻿ ? Т.к. ﻿
ϵ
∼
N
(
0
;
1
)
ϵ∼N(0;1)﻿ , то ﻿
y
i
∼
N
(
w
x
i
;
1
)
y 
i
​	
 ∼N(wx 
i
​	
 ;1)﻿ , т.к. это просто сдвиг на какое-то число. Запишем функцию правдоподобия для одного элемента выборки:
﻿
P
(
y
i
∣
x
i
,
w
,
ϵ
)
=
1
2
π
exp
⁡
{
−
(
y
i
−
x
i
w
)
2
2
}
P(y 
i
​	
 ∣x 
i
​	
 ,w,ϵ)= 
2π
​	
 
1
​	
 exp{− 
2
(y 
i
​	
 −x 
i
​	
 w) 
2
 
​	
 }﻿
Мы просто подставили в плотность нормального распределения наши величины. Возьмём от него натуральный логарифм:
﻿
ln
⁡
(
P
(
y
i
∣
x
i
,
w
,
ϵ
)
)
=
ln
⁡
(
1
2
π
exp
⁡
{
−
(
y
i
−
x
i
w
)
2
2
}
)
=
−
ln
⁡
2
π
2
−
1
2
(
y
i
−
x
i
w
)
2
ln(P(y 
i
​	
 ∣x 
i
​	
 ,w,ϵ))=ln( 
2π
​	
 
1
​	
 exp{− 
2
(y 
i
​	
 −x 
i
​	
 w) 
2
 
​	
 })=− 
2
ln2π
​	
 − 
2
1
​	
 (y 
i
​	
 −x 
i
​	
 w) 
2
 ﻿
Теперь запишем логарифм функции правдоподобия для всей выборки:
﻿
ln
⁡
P
(
Y
∣
X
,
w
,
ϵ
)
=
−
n
2
ln
⁡
2
π
−
∑
i
=
1
n
(
y
i
−
x
i
w
)
2
lnP(Y∣X,w,ϵ)=− 
2
n
​	
 ln2π− 
i=1
∑
n
​	
 (y 
i
​	
 −x 
i
​	
 w) 
2
 ﻿
Первое слагаемое можно отбросить, т.к. оно константа, а второе слагаемое что-то напоминает. Правильно, это MSE. Т.е. когда мы минимизируем ошибку MSE, то получившиеся веса ﻿
w
w﻿ будут также являться и оценкой максимального правдоподобия, и будут обладать всеми свойствами этой оценки (мы их опустили, но поверьте, обладать этими свойствами круто).
Поэтому эта функция ошибки часто используется в линейной регрессии.

Энтропия.

Энтропия используется не только в теории вероятностей, но и в физике. Но мы рассмотрим информационную энтропию. Это мера неопределённости некоторой системы. Например, пусть идёт поток бессвязных русских букв (твой ответ на экзамене). Все символы появляются равновероятно и энтропия данной системы максимальна. А вот если мы знаем, что идёт поток не
бессвязных букв, а слова русского языка. Энтропия уменьшается, т.к. мы можем делать некие оценки на то, какая буква появится следующей. А если будет идти поток не слов, а грамотно составленные тексты. Энтропия ещё сильнее уменьшится.

Дадим формальное определение энтропии. Пусть у нас есть случайный эксперимент ﻿
X
X﻿ с ﻿
n
n﻿ возможными ответами, т.е. ﻿
X
X﻿  —  дискретная случайная величина. Вероятность каждого ответа равна ﻿
p
i
p 
i
​	
 ﻿ . Тогда энтропия равна:
﻿
H
(
X
)
=
i
=
1
n
∑
i
=
1
n
p
i
2
p
i
H(X)= 
i=1
n
​	
  
i=1
∑
n
​	
 p 
i
​	
 log 
2
​	
 p 
i
​	
 ﻿

Кросс-энтропия.

Кросс-энтропия или перекрёстная энтропия показывает количественную разницу между двумя распределениями вероятностей. Она определяется следующим образом:
﻿
H
(
p
;
q
)
=
−
x
p
(
x
)
log
⁡
q
(
x
)
H(p;q)=− 
x
∑
​	
 p(x)logq(x)﻿
Эту функцию можно использовать как функцию потерь (её называют  логистической функцией потерь  ). Например, ﻿
q
q﻿  —  это истинное распределение, оно выглядит как куча нулей на неправильных классах, и одна единичка на правильном. А ﻿
p
p﻿  —  это результат нашего алгоритма, где мы говорим "вот к ﻿
1
1﻿ -му классу объект относится с вероятностью ﻿
0
,
2
0,2﻿ , а вот ко ﻿
2
2﻿ -му с вероятностью ﻿
0
,
7
0,7﻿ и т.д.". Эту функцию мы можем дифференцировать
и поэтому сможем использовать разные методы обучения.

Связь кросс-энтропии с методом максимального правдоподобия.

Давайте пример. Пусть у нас есть задача классификации на ﻿
2
2﻿ класса: ﻿
0
0﻿ и ﻿
1
1﻿ . В обучающей выборке ﻿
p
p﻿  —  это вероятность класса ﻿
1
1﻿ ,
а ﻿
1
−
p
1−p﻿  —  вероятность класса ﻿
0
0﻿ . Пусть у нас есть алгоритм
﻿
a
i
=
a
(
x
∣
w
)
a 
i
​	
 =a(x∣w)﻿ , т.е. он принимает на вход ﻿
x
x﻿ , смотрит на свои
внутренние веса ﻿
w
w﻿ и выдаёт ответ ﻿
a
i
∈
{
0
;
1
}
a 
i
​	
 ∈{0;1}﻿ .
Вероятность получить ответ ﻿
y
i
y 
i
​	
 ﻿ , если вход был ﻿
x
i
x 
i
​	
 ﻿ , а веса
﻿
w
w﻿ равна:
﻿
p
(
y
i
∣
x
i
,
w
)
=
a
i
y
i
(
1
−
a
i
)
1
−
y
i
p(y 
i
​	
 ∣x 
i
​	
 ,w)=a 
i
y 
i
​	
 
​	
 (1−a 
i
​	
 ) 
1−y 
i
​	
 
 ﻿
Давайте попробуем подобрать веса $w$ с помощью метода максимального правдоподобия:
﻿
log
⁡
{
p
(
y
∣
X
,
w
)
}
=
∑
i
=
1
n
log
⁡
{
p
(
y
i
∣
x
i
,
w
)
}
=
∑
i
=
1
n
log
⁡
{
a
i
y
i
(
1
−
a
i
)
1
−
y
i
}
=
∑
i
=
1
n
(
y
i
log
⁡
a
i
+
(
1
−
y
i
)
log
⁡
(
1
−
a
i
)
)
→
m
a
x
log{p(y∣X,w)}= 
i=1
∑
n
​	
 log{p(y 
i
​	
 ∣x 
i
​	
 ,w)}= 
i=1
∑
n
​	
 log{a 
i
y 
i
​	
 
​	
 (1−a 
i
​	
 ) 
1−y 
i
​	
 
 }= 
i=1
∑
n
​	
 (y 
i
​	
 loga 
i
​	
 +(1−y 
i
​	
 )log(1−a 
i
​	
 ))→max﻿
Последнее можно переписать, как:
﻿
−
∑
i
=
1
n
(
y
i
log
⁡
a
i
+
(
1
−
y
i
)
log
⁡
(
1
−
a
i
)
)
→
m
i
n
− 
i=1
∑
n
​	
 (y 
i
​	
 loga 
i
​	
 +(1−y 
i
​	
 )log(1−a 
i
​	
 ))→min﻿
Опа, а это же кросс-энтропия. Т.е. максимизация правдоподобия эквивалентна минимизации кросс-энтропии.
Т.е. для задачи линейной регрессии мы доказали, что логична MSE. А вот для задачи бинарной классификации оптимальна логистическая функция потерь.

Логистическая регрессия.

На основе полученных знаний можно сделать ещё один классификатор.
Есть волшебная функция  сигмоидная функция  : ﻿
f
(
z
)
=
1
1
+
e
−
z
f(z)= 
1+e 
−z
 
1
​	
 ﻿
Она позволит превратить любое число в число от 0 до 1. Ещё она хорошо связана с вероятностями классов ﻿
1
1﻿ и классов ﻿
0
0﻿ , но мы пока не будем касаться этой темы, поэтому выбрали её, а не какую-то другую.
Пусть у нас есть задача бинарной классификации. Мы хотим обучить модель, которая бы выдавала следующую вероятность ﻿
P
(
y
=
1
∣
x
)
P(y=1∣x)﻿ . Т.е. решать нашу задачу классификации.
Пусть у каждого элемента обучающей выборки есть ﻿
n
n﻿  —  признаков.
Давайте использовать линейную модель:
﻿
P
(
y
=
1
∣
x
)
=
w
0
+
w
1
x
1
+
w
2
x
2
+
⋯
+
w
n
x
n
P(y=1∣x)=w 
0
​	
 +w 
1
​	
 x 
1
​	
 +w 
2
​	
 x 
2
​	
 +⋯+w 
n
​	
 x 
n
​	
 ﻿
Проблема в том, что выражение слева должно быть от 0 до 1, а выражение справа может принимать любые действительные значения. Давайте преобразуем выражение слева. Рассмотрим функцию ﻿
o
d
d
s
(
p
)
=
p
1
−
p
odds(p)= 
1−p
p
​	
 ﻿ :

Т.е. мы будем моделировать не ﻿
P
(
y
=
1
∣
x
)
P(y=1∣x)﻿ ,
а ﻿
o
d
d
s
(
P
(
y
=
1
∣
x
)
)
=
P
(
y
=
1
∣
x
)
1
−
P
(
y
=
1
∣
x
)
odds(P(y=1∣x))= 
1−P(y=1∣x)
P(y=1∣x)
​	
 ﻿ .
Действительно, оценив эту величину, мы всё равно сможем сказать, какая вероятность объекту принадлежать к классу ﻿
1
1﻿ . Но проблема в том, что эта величина только положительная. Сделаем ещё однру трансформацию: ﻿
log
⁡
(
o
d
d
s
(
P
(
y
=
1
∣
x
)
)
log(odds(P(y=1∣x))﻿ :

Эта величина уже может принимать значения от ﻿
−
∞
 
д
о
+
∞
−∞ до+∞﻿ .
К тому же, она симметрична относительно ﻿
0
0﻿ . В итоге: ﻿
log
⁡
P
(
y
=
1
∣
x
)
1
−
P
(
y
=
1
∣
x
)
=
w
0
+
w
1
x
1
+
⋯
+
w
n
x
n
log 
1−P(y=1∣x)
P(y=1∣x)
​	
 =w 
0
​	
 +w 
1
​	
 x 
1
​	
 +⋯+w 
n
​	
 x 
n
​	
 ﻿

Если мы попробуем выразить из уравнения ﻿
P
(
y
=
1
∣
x
)
P(y=1∣x)﻿ , то как раз и получим сигмоидную функцию:
﻿
P
(
y
=
1
∣
x
)
=
1
1
+
e
−
w
0
−
w
1
x
1
−
…
=
f
(
w
0
+
w
1
x
1
+
…
)
P(y=1∣x)= 
1+e 
−w 
0
​	
 −w 
1
​	
 x 
1
​	
 −…
 
1
​	
 =f(w 
0
​	
 +w 
1
​	
 x 
1
​	
 +…)﻿

Это и есть то, что предсказывает логистическая регрессия.
В предыдущей части у нас был алгоритм классификации ﻿
a
(
x
∣
w
)
a(x∣w)﻿ . В нашем случае:
﻿
a
(
x
∣
w
)
=
f
(
w
0
+
w
1
x
1
+
⋯
+
w
n
x
n
)
a(x∣w)=f(w 
0
​	
 +w 
1
​	
 x 
1
​	
 +⋯+w 
n
​	
 x 
n
​	
 )﻿

Это и есть вся логистическая регрессия.
Обучают её с помощью той же кросс-энтропии, как описано в предыдущей теме, т.е.:
﻿
−
∑
i
=
1
n
(
y
i
log
⁡
a
i
+
(
1
−
y
i
)
log
⁡
(
1
−
a
i
)
)
→
m
i
n
− 
i=1
∑
n
​	
 (y 
i
​	
 loga 
i
​	
 +(1−y 
i
​	
 )log(1−a 
i
​	
 ))→min﻿
