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
            <nav class="site-header sticky-top py-1">
                <div class="container d-flex flex-column flex-md-row justify-\
                    content-between">
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
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
"""
