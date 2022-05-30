def add_TLS(action, fig):
    print(action)
    if action==0:
        colors=['green', 'red', 'red','red','green', 'red', 'red','red']
    if action==1:
        colors=['red', 'green', 'red','red','red', 'green', 'red','red']
    if action==2:
        colors=['red', 'red', 'green','red','red', 'red', 'green','red']
    if action==3:
        colors=['red', 'red', 'red','green','red', 'red', 'red','green']
    if action==4:
        colors=['green', 'green', 'red','red','red', 'red', 'red','red']
    if action==5:
        colors=['red', 'red', 'green','green','red', 'red', 'red','red']
    if action==6:
        colors=['red', 'red', 'red','red','green', 'green', 'red','red']
    if action==7:
        colors=['red', 'red', 'red','green','green', 'red', 'green','green']

        
    fig.add_shape(type="rect",
        x0=-750, y0=-9.8, x1=750, y1=0,
        line=dict(
            color="black",
            width=1,
        ),
    )
    fig.add_shape(type="rect",
        x0=-750, y0=0, x1=750, y1=9.8,
        line=dict(
            color="black",
            width=1,
        ),
    )
    fig.add_shape(type="rect",
        x0=-9.8, y0=750, x1=0, y1=-750,
        line=dict(
            color="black",
            width=1,
        ),
    )
    fig.add_shape(type="rect",
       x0=0, y0=750, x1=9.2, y1=-750,
        line=dict(
            color="black",
            width=1,
        ),
    )
    fig.add_vline(x=93.3-100, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=96.6-100, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=103-100, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=106-100, line_width=0.5, line_dash="dash", line_color="black")

    fig.add_hline(y=-6.2, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_hline(y=-3, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_hline(y=3, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_hline(y=6.2, line_width=0.5, line_dash="dash", line_color="black")

    fig.add_shape(type="line",
        x0=90.2-100, y0=9.8, x1=93.3-100, y1=9.8,
        line=dict(
            color=colors[0],
            width=4,
        )
    )
    fig.add_shape(type="line",
        x0=93.3-100, y0=9.8, x1=96.6-100, y1=9.8,
        line=dict(
            color=colors[0],
            width=4,
        )
    )
    fig.add_shape(type="line",
        x0=96.6-100, y0=9.8, x1=100-100, y1=9.8,
        line=dict(
            color=colors[1],
            width=4,
        )
    )

    fig.add_shape(type="line",
        x0=100-100, y0=-9.8, x1=103-100, y1=-9.8,
        line=dict(
            color=colors[5],
            width=4,
        )
    )
    fig.add_shape(type="line",
        x0=103-100, y0=-9.8, x1=106-100, y1=-9.8,
        line=dict(
            color=colors[4],
            width=4,
        )
    )
    fig.add_shape(type="line",
        x0=106-100, y0=-9.8, x1=109.2-100, y1=-9.8,
        line=dict(
            color=colors[4],
            width=4,
        )
    )


    fig.add_shape(type="line",
        x0=90.2-100, y0=0, x1=90.2-100, y1=-3,
        line=dict(
            color=colors[7],
            width=4,
        )
    )
    fig.add_shape(type="line",
        x0=90.2-100, y0=-3, x1=90.2-100, y1=-6.2,
        line=dict(
            color=colors[6],
            width=4,
        )
    )
    fig.add_shape(type="line",
        x0=90.2-100, y0=-6.2, x1=90.2-100, y1=-9.8,
        line=dict(
            color=colors[6],
            width=4,
        )
    )


    fig.add_shape(type="line",
        x0=109.2-100, y0=0, x1=109.2-100, y1=3,
        line=dict(
            color=colors[2],
            width=4,
        )
    )
    fig.add_shape(type="line",
        x0=109.2-100, y0=3, x1=109.2-100, y1=6.2,
        line=dict(
            color=colors[2],
            width=4,
        )
    )
    fig.add_shape(type="line",
        x0=109.2-100, y0=6.2, x1=109.2-100, y1=9.8,
        line=dict(
            color=colors[3],
            width=4,
        )
    )

    print(colors)

    return fig

    
