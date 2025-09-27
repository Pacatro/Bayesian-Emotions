import typer
from typing import Annotated

import config
from cli_commands import app as app_commands


app = typer.Typer(
    rich_markup_mode=None,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(app_commands)


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Verbose mode")
    ] = False,
    show_data_stats: Annotated[
        bool, typer.Option("--show-data-stats", "-s", help="Show data stats")
    ] = False,
):
    config.state["verbose"] = verbose
    config.state["show_data_stats"] = show_data_stats


if __name__ == "__main__":
    app()
