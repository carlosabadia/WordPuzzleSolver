### 1. Imports and class names setup ###
from ast import Interactive
import gradio as gr
import os
import argparse
import numpy as np
import os
import pytesseract
import re
import shutil
import solver
import glob
from PIL import Image
from typing import Tuple, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


# Setup class names
with open("class_names.txt", "r") as f:  # reading them in from class_names.txt
    class_names = [names.strip() for names in f.readlines()]


def get_images():
    images_list = []
    for filename in glob.glob('wordsPuzzle/*.jpg'):  # assuming png
        im = Image.open(filename)
        images_list.append(im)
    return images_list


def main():
    args = parse_args()
    print('*** Now using %s.' % (args.device))

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown('''# World Puzzle Solver 🧩''')
        gr.Markdown('''## (Works in Spanish too!) 🇪🇸''')

        with gr.Box():
            gr.Markdown(
                '''### Insert a Word Puzzle Image in both boxes and crop the board and words''')
            with gr.Row():
                with gr.Box():
                    with gr.Column():
                        gr.Markdown('''Images 🖼️''')
                        with gr.Row():
                            input_board = gr.Image(label='Board',
                                                   type='filepath',
                                                   interactive=True,
                                                   placeholder="examples/board_test.png")
                        with gr.Row():
                            crop_board_button = gr.Button('Crop Board ✂️')
                        with gr.Row():
                            input_words = gr.Image(label='Words',
                                                   type='filepath',
                                                   interactive=True, height="300px", width="300px",
                                                   placeholder="examples/words_test.png")
                        with gr.Row():
                            crop_words_button = gr.Button('Crop Words ✂️')
                        with gr.Row():
                            # Create examples list from "examples/" directory
                            paths = [["examples/" + example]
                                     for example in os.listdir("examples")]
                            example_images = gr.Dataset(components=([input_board]),
                                                        samples=[[path]
                                                                 for path in paths],
                                                        label='Image Examples (Drag and drop into both boxes) then crop using the tool button')

                with gr.Box():
                    with gr.Column():
                        gr.Markdown('''Cropped Images ✂️''')
                        with gr.Row():
                            cropped_board = gr.Image(label='Board Cropped',
                                                     type='filepath',
                                                     interactive=False, height="auto")
                            instyle = gr.Variable()
                        with gr.Row():
                            cropped_words = gr.Image(label='Words Cropped',
                                                     type='filepath',
                                                     interactive=False)
                            instyle = gr.Variable()
                        with gr.Row():
                            find_words_button = gr.Button('Find Words 🔍')
                        with gr.Row():
                            words_found = gr.Textbox(
                                label='Words detected (edit if wrong)', interactive=True, value='')
                        with gr.Row():
                            solve_button = gr.Button('Solve! 📝')

                with gr.Box():
                    with gr.Column():
                        gr.Markdown('''Solution ✅''')
                        with gr.Row():
                            board_solved = gr.Image(
                                type='filepath',
                                interactive=False)
                        with gr.Row():
                            show_words_board = gr.Button(
                                'Show words seperately 📝')
                        with gr.Row():
                            gallery = gr.Gallery(
                                label=None, show_label=True, elem_id="gallery"
                            ).style(grid=[4], height="auto")


                crop_board_button.click(fn=None,
                                        inputs=[input_board],
                                        outputs=[cropped_board])
                crop_words_button.click(fn=None,
                                        inputs=[input_words],
                                        outputs=[cropped_words])
                find_words_button.click(solver.get_words,
                                        inputs=cropped_words,
                                        outputs=words_found)
                solve_button.click(solver.solve_puzzle,
                                   inputs=[cropped_board, words_found],
                                   outputs=board_solved)

        example_images.click(fn=set_example_image,
                             inputs=example_images,
                             outputs=example_images.components)
        show_words_board.click(get_images, None, gallery)

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
