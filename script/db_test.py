


def main():
    from config.base_opts import parse_opts
    from core.util.database import DBHelper

    parser = parse_opts()
    args = parser.parse_args()

    db_helper = DBHelper(args)
    db_helper.make_table()
    


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
    
    main()