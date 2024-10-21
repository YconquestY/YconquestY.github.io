import { sidebar } from "vuepress-theme-hope";

export default sidebar({
    "/": [
        "", // home
        {
            text  : "Blog",
            icon  : "blog",
            prefix: "blog/",
            children: [
                {
                    text: "Computer system",
                    icon: "network-wired",
                    link: "sys/"
                },
                {
                    text: "Computer architecture",
                    icon: "microchip",
                    link: "arch/"
                },
                {
                    text: "Machine learning",
                    icon: "brain",
                    link: "ml/nerf/"
                },
                /*
                {
                    text: "Developement",
                    icon: "code",
                }
                */
            ]
        },
        {
            text: "About",
            icon: "address-card",
            link: "about.html" 
        },
        {
            text: "Presentation",
            icon: "person-chalkboard",
            link: "prez/"
        }
    ]
});