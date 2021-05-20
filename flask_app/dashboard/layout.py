html_layout = """
<!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
        </head>
        <body class="dash-template">
            <nav class="navbar expand-lg navbar-light bg-light">
                <div class="container fluid">
                    <a class="navbar-brand" href="/">NetFlowOpt</a>
                    <form class="d-flex">
                        <a class="btn btn-primary" href="/" role="button">\
                            Logout</a>
                    </form>
                </div>
            </nav>
            <div class="container">
            {%app_entry%}
            </div>
            <footer class="bd-footer">
                <div class="container">
                    <div class="row">
                        <div class="col-10">
                            <p>&copy; Michele Urbani - 2021</p>
                        </div>
                    </div>
                </div>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
"""
