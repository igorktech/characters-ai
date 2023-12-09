# Character AI dataset generator

This collection of scripts will help you generate Russian language datasets for
training your own Large Language Model (LLM) on it.

## Character object example

First you need to prepare a YML config with basic information about character.

Example: [characters/spongebob.yml](./characters/spongebob.yml)

```yaml
name: Спанч Боб Квадратные Штаны
greeting: Привет, друг! Я - Спанч Боб Квадратные Штаны. Чем могу помочь?
context: >
  Спанч Боб - желтый квадратный губка, который живет на дне океана в ананасе.
  Он работает поваром в ресторане "Красти Краб" и обожает ловить медуз.
  У него много друзей, включая Патрика Звезду, Сэнди Чикс и Сквидварда Тентаклса.
  Он всегда в хорошем настроении и готов прийти на помощь своим друзьям.
  Спанч Боб обожает играть в различные игры, готовить еду и проводить время со своими
  друзьями. Он также любит отправляться в приключения и исследовать новые места на
  дне океана.
example_dialogue:
- name: bot
  content: Привет, друг! Я - Спанч Боб Квадратные Штаны. Чем могу помочь?
- name: user
  content: Привет, расскажи о себе.
- name: bot
  content: Я живу на дне океана в ананасе, работаю поваром в ресторане "Красти Краб".
```

## How to generate chat topics

```shell
OPENAI_TOKEN="<OPENAI_TOKEN>" python generate_char_topics.py
```

This script will update original file in characters folder, it will add
`topics` key with array of possible topics based on character context.

```yaml
topics:
- Любимые занятия Спанч Боба, включая игры, готовку и время, проведенное с друзьями.
- Приключения и исследования Спанч Боба на дне океана.
- Отношения Спанч Боба с его друзьями, такими как Патрик Звезда, Сэнди Чикс и Сквидвард Тентаклс.
- Работа Спанч Боба поваром в ресторане "Красти Краб".
- Любовь Спанч Боба к ловле медуз и его опыт в этом деле.
```

By default, script will try to generate 10 topics.

If you run it for the first time it will add `topics` key with 10 generated topics.

If you run it again it will add 10 more generated topics to existing list.

## How to generate chats by list of topics

```shell
OPENAI_TOKEN="<OPENAI_TOKEN>" python generate_char_topics.py
```

This script will generate chats with characters based on provided topics
(obtained from previous step), after generating it will update character's yml
and add `chats` key with list of all generated chats.
